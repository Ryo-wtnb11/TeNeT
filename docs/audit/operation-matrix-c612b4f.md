# Current operation matrix at `d612869`

This is the Phase A capability census required by #9. The authority is TeNeT
`d6128696c01b85d8dffbaf805dc1a34fa7686a48`; current source and executable
tests are authoritative. Export presence, trait plausibility, historical
documentation and old benchmark output are not support evidence.

## Status

- **PROVED**: the canonical typed public path has a current executable test for
  this operation family and provider family.
- **MEASURED-GAP**: the path works, but current measurements prove a remaining
  production cost or reference mismatch.
- **UNSUPPORTED**: the current type/capability boundary does not expose the
  operation.
- **INTENTIONAL-DIFFERENCE**: TeNeT deliberately has a different public or
  ownership contract and records why.
- **NEEDS-PROOF**: the implementation is reachable by bounds or a nearby test,
  but this exact user-facing row is not independently proved.

`PROVED` is deliberately narrow: it does not mean every rank, sector
distribution or backend has been exhaustively tested.

## Provider families on default Host storage

All cells refer to `TensorMap<R, D, Vec<D>>`. `f64/c64` means payload dtype;
categorical coefficients are separately constrained by the provider bounds.

| Operation family | U(1), Z2 | ZN, CU(1) | fZ2 | SU(2) | multiplicity-free product | Fibonacci | checked Generic / SU(N) |
|---|---|---|---|---|---|---|---|
| Space and tensor construction; labelled block readback | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED | PROVED |
| Physical dense expansion and symmetric projection | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| `adjoint` | PROVED lazy view | NEEDS-PROOF | PROVED lazy view | PROVED lazy view | PROVED lazy view | UNSUPPORTED | PROVED for SU(3)/SU(4) real-provider fixtures |
| `permute`, `transpose`, `repartition` | PROVED | PROVED for CU(1); NEEDS-PROOF for ZN | PROVED | PROVED | PROVED | UNSUPPORTED | PROVED |
| `braid` | PROVED | NEEDS-PROOF | PROVED | PROVED | PROVED | UNSUPPORTED | PROVED |
| `twist` | PROVED | NEEDS-PROOF | PROVED | PROVED | PROVED | UNSUPPORTED | PROVED for synthetic checked-pivotal f64/c64 ±1 scaling, μ=2 full keys, direct/lazy parity, precedence, staged failures, NoBraiding/Bosonic/identity sharing, plus SU(3)/SU(4) × f64/c64 identity sharing and exact layout; checked-Generic compact construction remains unsupported |
| `flip` | PROVED | NEEDS-PROOF | PROVED | NEEDS-PROOF | NEEDS-PROOF | UNSUPPORTED | PROVED for synthetic checked-pivotal f64/c64 χ/θ scaling, μ=2 full keys, repeated legs, direct/lazy duality parity, validation precedence, staged failures, NoBraiding rejection, plus SU(3)/SU(4) × f64/c64 layout preservation and inverse roundtrips; checked-Generic compact construction remains unsupported |
| unit insertion/removal | PROVED | NEEDS-PROOF | PROVED | NEEDS-PROOF | NEEDS-PROOF | UNSUPPORTED | PROVED |
| `catdomain`, `catcodomain` | PROVED | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | UNSUPPORTED | PROVED in both directions for four SU(3)/SU(4) × f64/c64 cases with μ=2 full-key matching, exact direct-sum slabs, distinct equal-identity Arcs, and lazy-adjoint parity; compact checked-Generic construction remains unsupported |
| `absorb` | PROVED | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | UNSUPPORTED | PROVED for four SU(3)/SU(4) × f64/c64 cases with μ=2 vertex keys, asymmetric min-prefix copy, and lazy-adjoint parity; provider-wide coverage remains open |
| `otimes` | PROVED | NEEDS-PROOF | PROVED | PROVED | PROVED | UNSUPPORTED | PROVED |
| arbitrary `contract`, ordered output, `compose` | PROVED | NEEDS-PROOF | PROVED | PROVED | PROVED | UNSUPPORTED | PROVED |
| `add`, `scale`, `norm`, `inner`, `trace_pairs`, `tr` | PROVED | NEEDS-PROOF | PROVED | PROVED | PROVED | UNSUPPORTED | checked-Generic dispatch and current real/complex fixtures cover `add`/`scale`/`norm`/`inner`/`trace_pairs`/`tr`; provider-wide coverage remains open |
| compact QR and compact SVD | PROVED | NEEDS-PROOF | PROVED | PROVED | PROVED | UNSUPPORTED | PROVED on checked-Generic real/complex and SU(3)/SU(4) reconstruction fixtures |
| full/truncated SVD, values, LQ | PROVED on U1/Z2 fixtures | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | UNSUPPORTED | full/truncated SVD, values, compact/full QR/LQ proved on current checked-Generic fixtures |
| orthogonal factors (`left_orth`, `right_orth`) | PROVED on U1/Z2 fixtures | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | UNSUPPORTED | proved through checked-Generic compact QR/LQ parity fixtures |
| null/polar factors | PROVED on U1/Z2 fixtures | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | UNSUPPORTED | UNSUPPORTED |
| EIG/EIGH, `exp`, `inv`, left `solve`, `pinv`, `powi` | PROVED on Z2 fixtures | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | UNSUPPORTED | dense checked-Generic `inv`, global-cutoff `pinv`, Padé-only `exp`, integer `powi`, and left `solve` are proved. `pinv` has synthetic f64/c64 rectangular Moore-Penrose, invalid-cutoff precedence, strict global-boundary, exact-Arc/lazy-parent, staged SVD/GEMM failure counts, and SU(3)/SU(4) × f64/c64 μ=2 full-key coverage; TensorKit remains block-local by source audit. Left solve has synthetic preflight/admission counters, direct-into malformed-route coverage, private owned/lazy cache and five-route dense-call checks, backend provenance, and SU(3)/SU(4) × f64/c64 μ=2 full-key/cross-μ oracles; `solve_right` is unsupported. Compact checked-Generic construction remains unsupported. `eig_vals`/`eigh_vals` are proved, while other factor-returning spectra and matrix functions remain UNSUPPORTED |
| dense diagonal `sqrt` | PROVED on Z2 fixtures | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | UNSUPPORTED | checked-Generic dense diagonal `sqrt` proved on SU(3)/SU(4) f64/c64 SVD-bond fixtures; compact construction/preservation remains unsupported |
| typed network planning and execution | PROVED | NEEDS-PROOF | PROVED | PROVED | PROVED | UNSUPPORTED | UNSUPPORTED |
| serialization | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |

### Why Fibonacci is unsupported here

`FibonacciFusionRule` is a real expert-layer categorical implementation, not a
working canonical typed provider. It uses
`MultiplicityFreeFusionSymbols::Scalar = Complex64` and has no `SectorCodec`.
The multiplicity-free typed admission and operation dispatch require
`MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra +
SectorCodec` (`tenet/src/typed.rs:2786-3047`). Consequently a public re-export
does not establish `GradedSpace<FibonacciFusionRule>` construction or ordinary
typed operations. Current documentation must not list Fibonacci alongside the
proved typed providers without this qualification.

### Why checked Generic stops after core tensor operations

Provider-mode dispatch now covers transactional root construction, tree
transforms, `otimes`, `contract`, `compose`, reductions, trace, arithmetic, and
the current checked-Generic unit, cat, twist, decomposition, spectrum, and integer `powi` leaves.
The remaining operation gaps are factor-returning EIG/EIGH variants,
null/polar factors, matrix functions other than `exp` and integer `powi`, and typed network execution.
#662 closed through merged #946 after proving typed provider-error propagation
and transactional nonpublication for the reachable checked-Generic operations.

## Storage and device matrix

| Capability | Host `Vec<D>` multiplicity-free | Host `Vec<D>` checked Generic | other `S` with `TensorStorage` / `HostReadableStorage` | CUDA f64 multiplicity-free | CUDA f64 checked Generic | CUDA c64 |
|---|---|---|---|---|---|---|
| Metadata, provider ownership, handle clone | PROVED | PROVED | PROVED by representation bounds | PROVED | PROVED for transfer ownership | UNSUPPORTED |
| Stable `data() -> &[D]` | PROVED | PROVED | INTENTIONAL-DIFFERENCE: only `S: HostReadableStorage<D>` | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| Physical dense expansion and symmetric projection | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| Explicit Host/device transfer | N/A | N/A | NEEDS-PROOF for future storage types | PROVED | PROVED transfer-only | UNSUPPORTED |
| Adjoint | PROVED lazy view | PROVED lazy view for SU(3)/SU(4) real-provider fixtures | NEEDS-PROOF | PROVED lazy view | UNSUPPORTED | UNSUPPORTED |
| General permutation/braid/recoupling | PROVED | PROVED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| Canonical direct contract/compose | PROVED | PROVED | UNSUPPORTED | PROVED | UNSUPPORTED | UNSUPPORTED |
| Noncanonical transform-dependent contraction | PROVED | PROVED | UNSUPPORTED | UNSUPPORTED with typed preflight error | UNSUPPORTED | UNSUPPORTED |
| Arithmetic and reductions | PROVED | checked-Generic `add`/`scale`/`norm`/`inner`/`trace_pairs`/`tr` are proved on current fixtures; provider-wide matrix remains open | UNSUPPORTED | PROVED | UNSUPPORTED | UNSUPPORTED |
| QR | PROVED compact/full | checked-Generic compact/full proved on current Host fixtures | UNSUPPORTED | PROVED compact only | UNSUPPORTED | UNSUPPORTED |
| SVD | PROVED compact/full/truncated/values | checked-Generic compact/full/truncated/values proved on current Host fixtures | UNSUPPORTED | PROVED compact and truncated | UNSUPPORTED | UNSUPPORTED |
| EIGH | PROVED full/truncated/values | `eigh_vals` proved; factor-returning EIGH remains unsupported | UNSUPPORTED | PROVED full and truncated | UNSUPPORTED | UNSUPPORTED |
| LQ | PROVED on selected Host fixtures | checked-Generic compact/full LQ proved on current Host fixtures | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| orthogonal factors | PROVED on selected Host fixtures | checked-Generic `left_orth`/`right_orth` proved through compact QR/LQ parity fixtures | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| EIG, null/polar, solve and matrix functions | PROVED on selected Host fixtures | dense `inv`, dense Padé `exp`, integer `powi`, `eig_vals`, and left `solve` proved; `solve_right`, null/polar and remaining matrix functions remain unsupported; compact checked-Generic construction remains unsupported | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| Network planning/execution | PROVED | UNSUPPORTED | planning is storage-generic; Host execution is not | PROVED for canonical chains; trace/noncanonical scopes reject | UNSUPPORTED | UNSUPPORTED |

CUDA `PROVED` means the real-device tests exist and are explicitly ignored on
non-CUDA hosts. It is not evidence that default CI executes them. The gates are
`tenet/tests/typed_cuda_transfer.rs` and
`tenet-network/tests/typed_cuda_network.rs`.

## Current call-path authority

| Public family | Admission/lowering | Execution/publication |
|---|---|---|
| construction | `TypedTensorRootDispatch` selected by `R::Mode` | complete bound layout, then payload fill and publication |
| transforms | `TypedTensorTransformDispatch` | multiplicity-free completed transformer; checked Generic checked tree context |
| tensor product | `TypedTensorProductDispatch` | owned Host output retaining the left provider allocation |
| contraction/composition | `TypedTensorContractDispatch` | multiplicity-free or checked Generic typed lowering, then owned Host output |
| Host reductions/decomposition | multiplicity-free impl block at `tenet/src/typed.rs:8506` | `tenet-matrixalgebra` / Tenferro dense lease, then owned factors |
| CUDA operations | `TensorMap<R, f64, CudaStorage>` impl at `tenet/src/typed.rs:5172` | selected device plan/kernel; unsupported scopes reject before publication |
| network | multiplicity-free bounds in `tenet-network/src/network.rs` and `plancache.rs` | Runtime plan cache plus Host workspace, or narrow canonical CUDA replay |

The family summary above is not sufficient to prove the #9 execution-path
requirement by itself. The table below records every advertised operation row
at the boundary where its path actually differs. Rows marked `UNSUPPORTED` in
the capability tables have no call path and are intentionally omitted.

| Advertised row | Ownership and result layout | Provider/coefficient and planning path | Numeric execution | Publication and retained state |
|---|---|---|---|---|
| Host construction and block readback | `GradedSpace` and `TensorMap` retain the caller's `Arc<R>`; `TypedTensorRootDispatch::build_root` admits the final `FusionTreeHomSpace` before payload allocation | multiplicity-free and checked-Generic admission query their provider through separate mode dispatch; checked errors remain typed | `zeros`, `rand`, and `from_data` fill the complete admitted layout | one owned body is published only after admission and length validation; readback decodes labels through the retained provider |
| Host `adjoint` | swaps logical domain/codomain and stores the canonical parent in `TypedAdjointView` | no F/R/CGC query and no transform plan | no payload kernel at construction | publishes a parent-backed view; only `data()` may populate its receiver-owned `OnceLock` materialization |
| Host `permute`, `transpose`, `repartition`, `braid` | derives the destination from the source bound space; output retains the source provider allocation | multiplicity-free uses the Runtime tree-transform context/cache; checked Generic uses `tree_transform_dyn_owned_checked_generic_in_context` and its provider-bound checked context | tree-transform pack/replay/scatter; a multiplicity-free lazy adjoint lowers the operation to its parent, while checked Generic currently accepts owned data only | owned output after planning/execution succeeds; overwrite forms preserve the caller destination identity and allocation and do not publish a replacement body |
| Host `twist` | preserves the exact admitted space and provider allocation | mode dispatch keeps the multiplicity-free path unchanged; checked Generic validates index/empty/braiding in order and stages every fallible selected-sector twist before replay, with no provider query after staging | identity cases share the body; otherwise one fresh dense or compact scaled payload; a lazy adjoint redirects to its parent and reuses the already-admitted logical view | publishes only after staging and local scaling succeed; no root admission, prepared object, or result cache |
| Host `flip` | mode dispatch derives the toggled HomSpace and admits it through `TypedTensorRootDispatch` with the source `Arc`, then checks exact layout identity | multiplicity-free keeps its existing provider path; checked Generic rejects NoBraiding before pivotal queries and stages each unique FS/twist pair before replay; no tensor-result cache | one block-scaled output copy; lazy adjoint admits its toggled logical view, redirects to the parent with the inverse operation, and stays cold | fresh owned body only after admission, staging, and scaling; no result cache |
| Host unit insertion/removal | derives and validates the unit-leg HomSpace correspondence | canonical-unit provider marker and vacuum identity; no F/R/CGC plan | no dense copy for an owned dense input | new body shares the existing payload `Arc`; compact or lazy input may require one operation-local dense payload first |
| Host `catdomain`, `catcodomain` | derives the result HomSpace, then admits it with the left operand's exact provider `Arc` | mode root dispatch performs the one output admission; admitted input stamps reject identity mismatch, and no provider query follows successful output admission | compiled direct-sum slab copy; a conservatively declined lazy orientation retries locally after uncached materialization against the same admitted output | fresh owned output only after admission, planning and copy all succeed; operation-local temporaries are dropped and never enter the receiver `OnceLock` |
| Host `absorb` | preserves the destination admitted space and layout | compares the admitted layout rule-identity stamps; no provider or coefficient query and no retained prepared object | per-block min-prefix copy; lazy inputs may use `materialized_tensor_uncached()` | fresh owned output on the destination admitted space; operation-local temporaries are dropped and never enter the receiver `OnceLock` |
| Host `otimes` | result space is built from both bound input spaces and retains the left provider allocation | mode dispatch selects multiplicity-free or checked-Generic product authority | `tensorproduct_owned_multiplicity_free` or `tensorproduct_owned_checked_generic`; both currently materialize lazy inputs operation-locally | fresh owned Host body after the complete product succeeds |
| Host arbitrary `contract`, ordered contract, `compose` | validates runtime/provider/contracted spaces, derives final output order, then allocates the final bound output | mode dispatch selects multiplicity-free oriented lowering or checked-Generic owned lowering; tree/F/R work remains in the corresponding tensor-operation context | `tensorcontract_oriented_multiplicity_free` handles direct/lazy orientations; checked Generic calls `tensorcontract_owned_checked_generic` and rejects lazy inputs | fresh owned output after validation and execution; destination forms validate before replay and preserve destination identity |
| Host `add` and `scale` | require the same logical layout for addition; result space is the admitted input space | no new category plan; diagonal/dense representation is selected locally | compact-spectrum fast paths where legal, otherwise one mapped dense pass | fresh compact or dense owned body; no operation-result cache |
| Host `norm`, `inner` | validates compatible admitted layouts; scalar output has no result space | consumes existing block/layout weights; no F/R/CGC generation or retained plan | blockwise reduction over direct or oriented data | scalar return only; no published tensor or persistent workspace |
| Host `trace_pairs`, `tr` | validates pair geometry and derives the selected traced HomSpace before execution | trace structural admission uses the bound provider and current fusion layout | `tensortrace_fusion_dyn_owned_checked` or the scalar trace path | fresh owned tensor or scalar only after validation/execution succeeds |
| Host compact/full/truncated SVD and values | matrix-algebra factor plans derive final left, bond, and right bound spaces; truncation selects kept sector spectra before factor publication | no F/R/CGC query; dense backend leases are request-local | `svd_*_dyn`; direct adjoint-aware factor mappings avoid a receiver-sized adjoint input for compact/full/truncated/value paths | final factors are built directly as owned dense/compact bodies; no factorization-result cache |
| Host compact/full QR and LQ | matrix-algebra derives factor spaces from the input bound HomSpace | no coefficient plan or result cache | `qr_*_dyn`; a lazy adjoint uses one uncached logical payload, while LQ maps through QR and materializes final adjoint factors where required | owned factors published together after dense success; temporary logical payloads are request-local |
| Host orthogonal aliases | `left_orth`/`right_orth` inherit the preceding compact QR/LQ result spaces and authority unchanged | inherits compact QR/LQ; no additional F/R/CGC query | exactly the compact QR/LQ kernels through the mode-neutral aliases | same owned factors and publication boundary as compact QR/LQ |
| Host null/polar factors | result spaces come from their separate matrix-algebra contracts, including unmatched/disjoint completion | no F/R/CGC query | separate dense null/polar kernels; polar has adjoint-parent routes, while some null output conversions use uncached materialization | owned factors only after the full operation succeeds |
| Host EIG/EIGH and values | validates endomorphism/Hermiticity and derives eigenvector/bond spaces | no category coefficient plan | `eig_*_dyn` / `eigh_*_dyn`; lazy adjoints currently use an operation-local logical payload | owned factors or spectra; no implicit numerical-result cache |
| Host `exp`, `inv`, left `solve`, `pinv`, `sqrt`, `powi` | checked-Generic `inv` and dense `pinv` admit swapped outputs with the source `Arc`; left `solve` validates identity/exact codomain/isomorphism then admits `domain(A) <- domain(B)` with `A`'s exact `Arc`; `exp` validates exact endomorphism and reuses the input bound space; integer `powi` validates endomorphism, uses the admitted identity for zero, and uses one inverse for negative powers | `inv`, left `solve`, and `pinv` perform one typed output admission before allocation; `pinv` rejects invalid `rcond` before provider work, validates admitted identity/layout, then stages one SVD per nonempty sector and local GEMM | compact diagonal maps and `solve_right` remain multiplicity-free only; checked-Generic `inv` and left `solve` are dense sector solves, `pinv` is globally thresholded dense SVD, `exp` is dense Padé, and integer `powi` is exponentiation by squaring over compose/inv. TensorKit's tolerance is block-local, while TeNeT uses strict global `rcond * sigma_max`. Compact checked-Generic construction/preservation remains unsupported | direct `powi(1)` may be an owned body-sharing clone; other direct results are owned outputs. Lazy `pinv` follows `(A^H)^+ = (A^+)^H` through its parent and finishes owned/uncached; lazy inverse and negative powi redirect through `(A^-1)^H`; lazy exp and left solve use operation-local materialization; all leave the receiver cache cold |
| Host typed network planning/replay | `Network::plan` validates labels/spaces and compiles pairwise axes/output permutations; inputs retain their providers | Runtime plan cache owns topology/schedule entries, not provider coefficients; each contraction step delegates to the ordinary typed contraction path | `execute_with_workspace` reuses destination/intermediate buffers through overwrite forms when admitted | final owned tensor returned; idle typed workspaces may be retained under the Runtime-wide byte budget and are observable through plan-cache stats |
| CUDA transfer and lazy adjoint | device storage owns buffer and ordinal; Runtime owns execution resources | no category coefficient work during transfer/adjoint construction | explicit upload/download; adjoint remains an orientation over device parent storage | device or Host result is explicit; no hidden `data()` download and no Host whole-payload cache |
| CUDA arithmetic, reductions, canonical contraction/compose | operation-specific preflight derives/validates the device result layout; noncanonical scopes reject before kernel launch | multiplicity-free only; canonical direct geometry avoids a Host tree-transform plan | selected CUDA kernels/dense backend paths | fresh device output or scalar; no implicit Host transfer |
| CUDA QR/SVD/EIGH | device factor plan derives final device factor spaces before execution | multiplicity-free `f64` only; no provider coefficient cache is added | supported compact/truncated/full subset dispatches to the CUDA dense backend | owned device factors after success; unsupported variants are absent or reject before publication |
| CUDA network replay | reuses the compiled typed schedule only for canonical intermediate/final ordering | Runtime plan cache owns topology; ordinary CUDA contraction remains the numeric authority | `execute_cuda` chains canonical device contractions and rejects trace/noncanonical schedules before the first contraction | final device tensor; no Host workspace or transfer is introduced |

Two consequences are visible in this census. First, the checked-Generic path is
not merely missing facade methods: its existing transform/product path routes
through owned-data helpers, and contraction explicitly accepts only direct
owned inputs. The checked-Generic adjoint and direct reductions now have public
mode-dispatched paths, while operation coverage is still anchored to the
SU(3)/SU(4) real-provider fixtures listed above. Second, the Host multiplicity-free
decomposition surface does not retain factorization results or dense
workspaces; only network replay has an explicit bounded idle-workspace owner.
These are current facts, not recommendations to add caches.

Source anchors at the pinned revision are `tenet/src/typed.rs:2664-3221`
(provider-mode dispatch and oriented contraction), `:4358-5050` and `:6779`
(owned/diagonal/adjoint representation and uncached materialization),
`:5172-6690` (CUDA), `:7427-8495` (overwrite, transforms and contraction), and
`:8516-11728` (Host reductions, decompositions, matrix/index operations and
dtype conversion). Network planning/replay is anchored by
`tenet-network/src/network.rs:144-319,879-1110`; Runtime plan/workspace
retention by `tenet-network/src/plancache.rs:611-700,1110-1275`.

## Lazy-adjoint storage and execution

These are three different mechanisms and must not be combined under one
"adjoint cache" label:

| Mechanism | Current owner and trigger | Current scope |
|---|---|---|
| parent-backed lazy view | `TypedAdjointView` created by multiplicity-free Host/CUDA or checked-Generic Host `adjoint()` | metadata and canonical parent storage only; checked-Generic coverage is currently pinned by SU(3)/SU(4) real-provider fixtures |
| receiver-retained whole logical payload | the lazy view's `OnceLock`, populated only by Host-readable `data()` | compatibility storage for a stable `&[D]`, not an execution cache |
| operation-local full logical payload | `materialized_tensor_uncached()` | used only where no orientation-aware seam/algebraic redirect exists; never published to the receiver |

Current orientation-aware or algebraic routes avoid a receiver-sized input
copy for tree transforms, twist, contraction/composition, SVD compact/full/truncated/
values and pseudo-inverse. QR, some LQ/null output conversion, EIG/EIGH,
general `exp`, solve, `sqrt`, `absorb`, dtype conversion and conservative cat
fallbacks still contain operation-local full-payload calls
(`tenet/src/typed.rs`, all `materialized_tensor_uncached()` call sites). The
completed #783 review records this distinction. A future production leaf needs
a per-operation semantic proof and measured gap; a cold receiver cache alone
does not imply zero peak copy cost.

## Executable evidence

- Generic construction, multiplicity-two transforms, `otimes`, contraction,
  composition, typed failures and SUN round trips:
  `tenet/tests/checked_generic_facade.rs`.
- Checked-pivotal twist staging, ±1 full-key scaling, direct/lazy parity,
  NoBraiding/Bosonic behavior, typed failures, and SU(3)/SU(4) identity sharing:
  `tenet/tests/checked_generic_twist.rs`.
- U(1), SU(2), fZ2 and triple-product contraction laws plus U(1)/SU(2)
  TensorKit invariant streams: `tenet/tests/semantic_suite.rs`.
- Broad Host facade behavior, including CU(1) recoupling and Z2
  factorization/matrix-function fixtures: `tenet/tests/typed_facade.rs`.
- Product reductions and allocation contracts:
  `tenet/tests/inner_norm_allocations.rs` and
  `tenet/tests/permute_overwrite_allocations.rs`.
- Real-device transfer, arithmetic, reductions, QR, SVD, EIGH, canonical
  contraction and explicit unsupported scopes:
  `tenet/tests/typed_cuda_transfer.rs`.
- Canonical CUDA network execution and trace/noncanonical rejection:
  `tenet-network/tests/typed_cuda_network.rs`.

## Findings and routing

1. **Fibonacci public support is overstated.** The provider is expert-layer
   only under the current scalar/codec boundary. Route the user-facing claim
   correction through #9; a future implementation leaf must first decide how
   categorical coefficient scalar and payload scalar compose.
2. **Checked Generic is deliberately partial, not an alternate complete tensor
   hierarchy.** Close the residual factor-returning EIG/EIGH,
   null/polar, matrix-function, and network-execution gaps before
   describing SU(N) as generally supported. Merged #946 closed #662 with the
   shared typed error/nonpublication contract proved for the reachable checked
   operations.
3. **Storage genericity is representation-only today.** Metadata and host
   readback bounds are generic, but ordinary result allocation and execution
   remain Host `Vec<D>` or the explicit narrow CUDA impl. This matches #729's
   staged contract and is not itself a regression.
4. **CUDA rustdoc contains false negative examples.** The compile-fail examples
   around `tenet/src/typed.rs:5008` use `U1FusionRule` while claiming to prove
   checked-Generic CUDA methods are absent; U1 satisfies the multiplicity-free
   CUDA impl. Feature-enabled doctests must either use a checked-only provider
   or be replaced by a compile-time capability test.
5. **Per-provider proof is uneven.** U1/Z2, fZ2, SU2 and product rules have broad
   semantic gates. ZN and CU(1) mostly prove selected transfer/transform rows;
   do not upgrade the remaining cells from `NEEDS-PROOF` merely because the
   shared trait impl compiles.
6. **No canonical trivial/dense provider is exposed.** A one-sector Abelian
   tensor can serve as a fixture, but it is not a public no-symmetry dense
   tensor contract. Treat trivial/dense support as `UNSUPPORTED` until an
   actual public path is identified or added.

The next Phase A step is to turn every `NEEDS-PROOF` row that affects advertised
support into a minimal current-API smoke program, then reconcile the resulting
gaps with existing issues before opening new leaves.
