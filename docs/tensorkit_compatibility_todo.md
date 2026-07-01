# TensorKit Compatibility TODO

This file records TensorKit ecosystem behavior that TeNeT intentionally mirrors
or leaves unsupported until the categorical semantics are fixed. These items are
not optimization notes; they are semantic boundaries for the TensorKit-compatible
baseline.

## AdjointTensorMap with explicit braid

TensorKit source:

- `/Users/ryowatanabe/.julia/packages/TensorKit/6Camk/src/tensors/indexmanipulations.jl:163`
- `# TODO: braid for AdjointTensorMap; think about how to map the levels argument.`

TeNeT status:

- TeNeT already models explicit braid as `TreeTransformOperationKey::Braid`
  with axis permutations plus separate codomain/domain levels. The unresolved
  boundary is only lazy source adjoint lowering for that explicit braid.
- Ordinary lazy-adjoint lowering for `tensoradd`, `tensortrace`, and
  `tensorcontract` is implemented by swapping the categorical source space,
  remapping axes with `adjoint_tensor_axis`, and carrying storage conjugation
  separately.
- For symmetric braiding, TensorKit's `braid` path short-circuits to
  adjoint-aware `permute`, so this TODO does not block Z2, FermionParity, or
  ordinary symmetric tensoradd/contract fixtures.
- TensorKit leaves this undefined. TeNeT defines it as an extension; it is not
  part of the current TensorKit-compatible baseline.
- TeNeT implements this as a gated extension. Public fusion tensoradd only
  enables `Braid + source_conjugate` when the rule reports
  `supports_unitary_braid_dagger()`. Rules without that capability still return
  `UnsupportedTreeTransformScope`.

TeNeT extension definition:

- This extension is only mathematically valid for dagger-compatible braided
  fusion categories where the braiding is unitary:
  `c_{a,b}^\dagger = c_{a,b}^{-1}`. If a rule cannot provide this semantics,
  `Braid + source_conjugate` must stay unsupported.
- A lazy source adjoint first replaces the source tensor map by the categorical
  adjoint space view, including the codomain/domain swap. TeNeT's existing
  0-based adjoint axis map is used:
  `axis < NOUT ? NIN + axis : axis - NOUT`.
- Braid levels are attached to source strands, not to output tuple positions.
  `TreeTransformOperationKey::Braid` stores them split by source
  codomain/domain axes, matching TensorKit's `add_braid!` split by
  `codomainind(tsrc)` and `domainind(tsrc)`. Lowering remaps those split source
  levels with `adjoint_tensor_axis` and carries each strand level with it.
- The adjoint of a non-symmetric braid reverses the crossing direction. In
  TeNeT's level encoding this is represented by reversing the relative level
  order:
  `level' = min_level + max_level - level`.
  This turns every TensorKit-style `levels[i] < levels[j]` crossing into the
  inverse `levels'[i] > levels'[j]` crossing, and conversely.
- TeNeT core already represents double-tree braid levels as
  `codomain_levels ++ reverse(domain_levels)`, matching TensorKit's
  `fsbraid` construction. The extension must preserve that representation
  after adjoint-axis remapping and level-order reversal.

Implemented test coverage:

- `tenet-core` verifies that reflecting levels with
  `level' = min_level + max_level - level` selects the inverse Artin branch for
  a direction-sensitive anyonic rule.
- `tenet-operations` verifies source-strand level remapping separately from
  output tuple positions.
- `tenet-operations` validates bad braid axes, level count mismatch, and
  duplicate levels at lowering time.
- Public `tensoradd_fusion_into` rejects `Braid + source_conjugate` when the
  rule does not declare unitary dagger-compatible braiding.
- A supported unitary phase anyonic fixture compares public
  `source_conjugate + explicit Braid` against the explicit categorical
  sequence: adjoint-space view, inverse-level braid, then storage conjugation.

Remaining coverage to add before broadening beyond tensoradd:

- Cover codomain-only, domain-only, and mixed codomain/domain double-tree maps
  for all public operations that expose explicit braid with source conjugation.
- Include nonscalar SU2 recoupling and nontrivial degeneracy shapes once that
  operation path is enabled beyond tensoradd.
- Keep symmetric `Permute` behavior separate from explicit `Braid`; symmetric
  braid short-circuiting must not hide mistakes in non-symmetric braid lowering.

## Fermionic tensor trace vs ordinary trace

TensorKit source:

- `TensorOperations.@tensor` lowers repeated indices to `tensortrace!`.
- TensorKit `tensortrace!` routes to `trace_permute!`, which applies categorical
  twists for fermionic sectors.
- `LinearAlgebra.tr(::TensorMap)` is not the same operation; it sums block traces
  with dimension factors and does not represent the `@tensor` supertrace path.

TeNeT status:

- Plain dense `tensortrace` remains dense/TensorOperations-like.
- Fusion `tensortrace_fusion_*` implements categorical trace factors and
  fermionic supertrace behavior.
- Plain `tensortrace` rejects fusion-tree tensors so categorical trace is not
  silently replaced by dense trace.

## TensorOperations categorical lowering coverage

TensorKit source:

- TensorOperations exposes `tensorcontract!(C, A, pA, conjA, B, pB, conjB,
  pAB, ...)`; `conjA` and `conjB` are independent API states.
- TensorKit lowers `conjA`/`conjB` lazily by replacing the selected source with
  `A'`/`B'` and remapping axes with `adjointtensorindices`.
- For rank `(1, 1)`, TensorKit's 1-based adjoint axis map swaps `1 <-> 2`.
  TeNeT's 0-based map swaps `0 <-> 1`.

TeNeT fixed coverage:

- Dense `tensoradd`, `tensortrace`, and `tensorcontract` keep TensorOperations
  dense semantics and reject fusion-tree tensors instead of silently using the
  wrong categorical operation.
- Fusion `tensoradd` has lazy source-adjoint lowering and context replay tests.
- Fusion `tensortrace` has fermionic supertrace, degeneracy diagonal, lazy
  adjoint, beta-once, and SU2 quantum-dimension coverage.
- Fusion `tensorcontract` has lazy `lhs_conj`, `rhs_conj`, and `both_conj`
  coverage for scalar fusion blocks.
- Fusion `tensorcontract` has Z2 and FermionParity 2x2 degeneracy block
  coverage for:
  - `lhs_conj`: per sector `C_s = A_s† * B_s`
  - `rhs_conj`: per sector `C_s = A_s * B_s†`
  - `both_conj`: per sector `C_s = A_s† * B_s†`
- FermionParity W <- W no-braiding tests intentionally assert no extra
  fermionic twist for these three matrix-contract fixtures. TensorKit only
  inserts the tensorcontract twist when the effective contracted `B` leg is
  dual.
- Fusion `tensorcontract` has a noncanonical SU2 source-transform fixture with
  lazy `conjA && conjB`. The Rust test compares explicit-plan execution against
  the literal lowered sequence: source `tensoradd_fusion_into(...,
  source_conjugate=true)` transforms, then canonical contraction.
- The Julia oracle script records the corresponding TensorKit direct
  `tensorcontract!` result for the SU2 fixture. TeNeT's internal fusion-tree
  block storage is not flat-order identical to TensorKit's `C.data` for that
  multi-tree SU2 case, so the Rust assertion uses the TeNeT explicit reference
  sequence rather than direct flat-data equality.
- Fusion `tensorcontract` has a ProductSector(FermionParity x U1 x SU2)
  component-channel fixture with SU2 recoupling and complex data. This fixture
  has scalar degeneracy blocks, so the Rust data is asserted directly against
  the TensorKit oracle values.

Remaining semantic boundary:

- Extend the Julia TensorKit oracle script
  `tenet-operations/benchmarks/tensorkit_contract_adjoint_oracles.jl` as new
  Z2/FermionParity 2x2 degeneracy fixtures are added.
- Do not silently lower explicit non-symmetric `Braid + source_conjugate` as an
  ordinary `permute`; that is categorically wrong for non-symmetric braiding.
  Keep the TeNeT extension gated by explicit rule capability.
