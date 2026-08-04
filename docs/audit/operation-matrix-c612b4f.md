# Current operation matrix at `c612b4f`

This is the Phase A capability census required by #9. The authority is TeNeT
`c612b4ff5c0b9285fa6ef8617b5c5472cba3287e`; current source and executable
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
| `adjoint` | PROVED lazy view | NEEDS-PROOF | PROVED lazy view | PROVED lazy view | PROVED lazy view | UNSUPPORTED | UNSUPPORTED |
| `permute`, `transpose`, `repartition` | PROVED | PROVED for CU(1); NEEDS-PROOF for ZN | PROVED | PROVED | PROVED | UNSUPPORTED | PROVED |
| `braid` | PROVED | NEEDS-PROOF | PROVED | PROVED | PROVED | UNSUPPORTED | PROVED |
| `twist` | PROVED | NEEDS-PROOF | PROVED | PROVED | PROVED | UNSUPPORTED | UNSUPPORTED |
| `flip`, unit insertion/removal | PROVED | NEEDS-PROOF | PROVED | NEEDS-PROOF | NEEDS-PROOF | UNSUPPORTED | UNSUPPORTED |
| `catdomain`, `catcodomain`, `absorb` | PROVED | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | UNSUPPORTED | UNSUPPORTED |
| `otimes` | PROVED | NEEDS-PROOF | PROVED | PROVED | PROVED | UNSUPPORTED | PROVED |
| arbitrary `contract`, ordered output, `compose` | PROVED | NEEDS-PROOF | PROVED | PROVED | PROVED | UNSUPPORTED | PROVED |
| `add`, `scale`, `norm`, `inner`, `trace_pairs`, `tr` | PROVED | NEEDS-PROOF | PROVED | PROVED | PROVED | UNSUPPORTED | UNSUPPORTED |
| compact QR and compact SVD | PROVED | NEEDS-PROOF | PROVED | PROVED | PROVED | UNSUPPORTED | UNSUPPORTED |
| full/truncated SVD, values, LQ, orthogonal/null/polar factors | PROVED on U1/Z2 fixtures | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | UNSUPPORTED | UNSUPPORTED |
| EIG/EIGH, `exp`, `inv`, `solve`, `pinv`, `sqrt`, `powi` | PROVED on Z2 fixtures | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | UNSUPPORTED | UNSUPPORTED |
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

Provider-mode dispatch exists for transactional root construction, tree
transforms, `otimes`, `contract` and `compose`
(`tenet/src/typed.rs:2824-3087`). Reductions, factorizations, matrix functions,
unit/cat operations and network execution are in impl blocks requiring the
multiplicity-free `Scalar = f64` contract. This is the concrete residual scope
of #640 and #662, not a documentation-only gap.

## Storage and device matrix

| Capability | Host `Vec<D>` multiplicity-free | Host `Vec<D>` checked Generic | other `S` with `TensorStorage` / `HostReadableStorage` | CUDA f64 multiplicity-free | CUDA f64 checked Generic | CUDA c64 |
|---|---|---|---|---|---|---|
| Metadata, provider ownership, handle clone | PROVED | PROVED | PROVED by representation bounds | PROVED | PROVED for transfer ownership | UNSUPPORTED |
| Stable `data() -> &[D]` | PROVED | PROVED | INTENTIONAL-DIFFERENCE: only `S: HostReadableStorage<D>` | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| Explicit Host/device transfer | N/A | N/A | NEEDS-PROOF for future storage types | PROVED | PROVED transfer-only | UNSUPPORTED |
| Adjoint | PROVED lazy view | UNSUPPORTED | NEEDS-PROOF | PROVED lazy view | UNSUPPORTED | UNSUPPORTED |
| General permutation/braid/recoupling | PROVED | PROVED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| Canonical direct contract/compose | PROVED | PROVED | UNSUPPORTED | PROVED | UNSUPPORTED | UNSUPPORTED |
| Noncanonical transform-dependent contraction | PROVED | PROVED | UNSUPPORTED | UNSUPPORTED with typed preflight error | UNSUPPORTED | UNSUPPORTED |
| Arithmetic and reductions | PROVED | UNSUPPORTED | UNSUPPORTED | PROVED | UNSUPPORTED | UNSUPPORTED |
| QR | PROVED compact/full | UNSUPPORTED | UNSUPPORTED | PROVED compact only | UNSUPPORTED | UNSUPPORTED |
| SVD | PROVED compact/full/truncated/values | UNSUPPORTED | UNSUPPORTED | PROVED compact and truncated | UNSUPPORTED | UNSUPPORTED |
| EIGH | PROVED full/truncated/values | UNSUPPORTED | UNSUPPORTED | PROVED full and truncated | UNSUPPORTED | UNSUPPORTED |
| EIG, LQ, null/polar, solve and matrix functions | PROVED on selected Host fixtures | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
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

## Lazy-adjoint storage and execution

These are three different mechanisms and must not be combined under one
"adjoint cache" label:

| Mechanism | Current owner and trigger | Current scope |
|---|---|---|
| parent-backed lazy view | `TypedAdjointView` created by multiplicity-free Host/CUDA `adjoint()` | metadata and canonical parent storage only; checked Generic has no public `adjoint()` |
| receiver-retained whole logical payload | the lazy view's `OnceLock`, populated only by Host-readable `data()` | compatibility storage for a stable `&[D]`, not an execution cache |
| operation-local full logical payload | `materialized_tensor_uncached()` | used only where no orientation-aware seam/algebraic redirect exists; never published to the receiver |

Current orientation-aware or algebraic routes avoid a receiver-sized input
copy for tree transforms, contraction/composition, SVD compact/full/truncated/
values and pseudo-inverse. QR, some LQ/null output conversion, EIG/EIGH,
general `exp`, solve, `sqrt`, `absorb`, dtype conversion and conservative cat
fallbacks still contain operation-local full-payload calls
(`tenet/src/typed.rs`, all `materialized_tensor_uncached()` call sites). This is
the concrete audit list for #783; a cold receiver cache does not imply zero
peak copy cost.

## Executable evidence

- Generic construction, multiplicity-two transforms, `otimes`, contraction,
  composition, typed failures and SUN round trips:
  `tenet/tests/checked_generic_facade.rs`.
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
   hierarchy.** Complete reductions/decompositions under #640 and network
   execution under #662 before describing SU(N) as generally supported.
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
