# Writing a symmetry provider

A symmetry in TeNeT is a value, not a variant. There is no provider enum and no
symmetry-named branch inside any operation: a *provider* supplies the fusion
category — channels, duals, F/R symbols, quantum dimensions — and the engine
runs every tensor operation on top of it. Implementing the traits below in your
own crate gives you the same `GradedSpace<R>` / `TensorMap<R, D, S>` API the
built-in providers use.

This document is the contract. It says what you must implement, what each trait
owns, which laws the engine assumes without checking, and where an external
provider is still second-class today.

Read [`tenet/src/mathematics.md`](../tenet/src/mathematics.md) first if you need
the tensor-map convention itself (`codomain <- domain`, duality, block layout).
That document defines the semantics; this one defines the seam.

## The shape of the job

| step | trait | you supply |
| --- | --- | --- |
| 1 | `FusionRule` | identity, styles, vacuum, dual, fusion channels |
| 2 | `CheckedFusionAlgebra` | the same algebra, fallibly, over your representable domain |
| 3 | `SectorCodec` | your public label type and its encoding |
| 4 | `MultiplicityFreeFusionRule` | a marker: `N_ab^c ∈ {0, 1}` |
| 5 | `MultiplicityFreeFusionSymbols` | coefficient scalar, F, R |
| 6 | `MultiplicityFreeRigidSymbols` | dimensions, twist, Frobenius–Schur |
| 6b | `CanonicalUnitFusionRule` | a marker: your vacuum obeys the unit laws |

Steps 1–3 make spaces and tensors constructible. Steps 4–6 make index
manipulation, contraction, and factorization work. A provider with outer
multiplicity (`N_ab^c > 1`) uses a different set — see
[Outer multiplicity](#outer-multiplicity).

`tenet-sectors/src/fibonacci.rs` is the smallest complete non-abelian example;
`tenet-sectors/src/abelian.rs` has the abelian ones;
`tenet-sectors/src/su2.rs` shows a provider that delegates its coefficients to an
external crate.

## 1. `FusionRule`

`tenet-sectors/src/algebra.rs:39`. The engine's hot path. Every method is
infallible.

```rust
fn rule_identity(&self) -> RuleIdentity;
fn fusion_style(&self) -> FusionStyleKind;      // Unique | Simple | Generic
fn braiding_style(&self) -> BraidingStyleKind;  // NoBraiding | Bosonic | Fermionic | Anyonic
fn vacuum(&self) -> SectorId;
fn dual(&self, sector: SectorId) -> SectorId;   // defaults to identity
fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec;
```

**`SectorId` is opaque and provider-local.** It is an index your codec assigns,
not a physical charge and not a portable label. Two providers may use the same
numeric value for unrelated sectors; nothing in the type system prevents mixing
them, so the engine relies on `RuleIdentity` instead (below).

**The infallible methods carry a domain precondition.** They may panic on an id
your provider cannot represent. The engine only ever passes back ids it obtained
from your own `vacuum`, `dual`, or `fusion_channels`, so this is not a
user-facing hazard — but it does mean the validating twin in step 2 is where the
real domain check lives.

**Styles are claims the engine acts on.** `Unique` means every product has
exactly one channel; `Simple` means several channels but no multiplicity;
`Generic` means multiplicity. Declaring `Unique` when a product has two channels
selects a lowering that silently drops terms. `Bosonic`/`Fermionic` assert
symmetric braiding (`R` squares to the identity up to signs); `Anyonic` does not.

**`fusion_channels` order is part of your convention.** Keep it stable and match
your reference — `FibonacciFusionRule` documents that it lists the vacuum first
to match TensorKitSectors' iteration order (`fibonacci.rs:61-64`).

### `RuleIdentity`

`tenet-sectors/src/rule_identity.rs`. This is the cache key for every retained
layout, plan and coefficient in the engine. Equal identities are a promise that
*all* categorical data agrees: styles, vacuum, channels, duals, F, R,
dimensions, twists, bends, **and the codec**.

Three constructors, pick by how your provider carries its data:

- `RuleIdentity::of_type::<Self>()` — the provider is a unit struct with no
  data. All built-in group providers use this.
- `RuleIdentity::from_canonical_bytes::<Self>(prehash, bytes)` — the provider
  carries tables. Serialize everything that can change an answer.
- `RuleIdentity::new_unique::<Self>()` — you cannot canonicalize the data. Each
  allocation becomes its own identity, so nothing is shared between two
  instances that are in fact equal. Correct, and slower.

Do not use `TypeId` or the provider's address directly: one Rust type can hold
different tables, and two allocations of the same immutable rule must be able to
share cache entries.

## 2. `CheckedFusionAlgebra`

`tenet-sectors/src/algebra.rs:236`. The same three algebra queries, returning
`Result<_, FusionAlgebraError>`, and this is where you validate that an incoming
`SectorId` is in your domain:

```rust
fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError>;
fn try_fusion_channels(&self, l: SectorId, r: SectorId) -> Result<SectorVec, FusionAlgebraError>;
fn try_nsymbol(&self, l: SectorId, r: SectorId, c: SectorId) -> Result<usize, FusionAlgebraError>;
```

Transactional admission runs through these: an invalid or unrepresentable
algebra must publish no layout and no cache entry.

Today you write the algebra twice — once here and once infallibly in step 1 —
and the two must agree. In the shipped providers the infallible body is
literally the checked body with `?` replaced by `expect`
(`abelian.rs:620-664`). That duplication is tracked; write the checked body
first and make the infallible one forward to it.

## 3. `SectorCodec`

`tenet-sectors/src/algebra.rs:293`. Your public label type and its translation
to `SectorId`. Without it the provider still works at the expert layer but is
not reachable through `GradedSpace`/`TensorMap`, because those return labels,
never ids. (`FibonacciFusionRule` currently has no codec, which is exactly why
it is expert-only.)

Four laws:

1. `decode(encode(x)) == x` for every representable label;
2. `encode(decode(id)) == id` for every decodable id;
3. distinct labels never encode to the same id;
4. **decode is total over the reachable algebra**: `vacuum`, and every id your
   own `dual` and `fusion_channels` produce from decodable inputs, must decode.

Law 4 is the one that bites. A decode domain narrower than that closure is a
codec bug, not a user error: the engine hands you back exactly the ids your own
fusion produced, so a facade inspecting a legitimately built tensor would fail
on your own output. Encoding may still reject labels outside your representable
range — that value came from the user.

## 4–6. Coefficients

`MultiplicityFreeFusionRule` (`algebra.rs:320`) is an empty marker asserting
`N_ab^c ∈ {0, 1}`.

`MultiplicityFreeFusionSymbols` (`algebra.rs:322`) declares the coefficient
scalar and the two symbols:

```rust
type Scalar: Clone + Send + Sync;   // f64 or Complex64 in practice
fn f_symbol_scalar(&self, a: SectorId, b: SectorId, c: SectorId,
                   d: SectorId, e: SectorId, f: SectorId) -> Self::Scalar;
fn r_symbol_scalar(&self, a: SectorId, b: SectorId, c: SectorId) -> Self::Scalar;
```

Argument order is TensorKitSectors' `Fsymbol(a,b,c,d,e,f)` / `Rsymbol(a,b,c)`
verbatim: `e` is the `(a,b)` channel, `f` is the `(b,c)` channel, `d` is the
total. Return the additive zero for a disallowed configuration rather than
panicking — the engine prunes on it.

`MultiplicityFreeRigidSymbols` (`algebra.rs:360`) adds `dim`, `inv_dim`,
`sqrt_dim`, `inv_sqrt_dim`, `twist`, and the Frobenius–Schur phase. `a_symbol`
and `b_symbol` have correct default bodies derived from F; override only with a
reference to justify it.

Two capability flags, both defaulting to the conservative answer:

- `has_trivial_associator_gauge()` (`algebra.rs:339`) — return `true` only if
  *every* allowed F coefficient in your gauge is exactly the scalar unit. It
  selects a direct permutation lowering. Symmetric braiding alone is not enough.
- `supports_unitary_braid_dagger()` (`algebra.rs:60`).

`CanonicalUnitFusionRule` (`algebra.rs:125`) asserts the vacuum is self-dual,
that `1 ⊗ a -> a` and `a ⊗ 1 -> a` are the only channels involved, and that
unitors and associators act as the identity there. `FibonacciFusionRule` shows a
`Simple` provider that still qualifies (`fibonacci.rs:282-312`).

## Outer multiplicity

If `N_ab^c` can exceed 1, implement `CheckedGenericFusion` (`algebra.rs:527`)
and `CheckedGenericRigidSymbols` (`algebra.rs:652`) instead of steps 2/5/6. They
are deliberately independent of `FusionRule` so that a provider with a finite
generated catalog can reject an unavailable lookup without first manufacturing
an infallible answer. F and R become dense blocks (`GenericFArray`,
`GenericRMatrix`) indexed by one-based vertex labels.

`InfallibleGeneric<'a, R>` (`algebra.rs:692`) adapts an existing infallible
provider to the checked interface; see `tenet-sectors/src/sun.rs` for a
catalog-bounded provider.

Operator coverage on that path is still narrower than on the multiplicity-free
path.

## Products

`ProductFusionRule<Left, Right, Codec>` is itself a provider, so products nest.
The product's coefficient scalar is the promotion of its components'
(`PromoteCoefficientScalar`), following TensorKitSectors'
`fusionscalartype(ProductSector)`: components need not agree, and a complex
component widens the product. Use `PackedProductCodec` (fixed-width,
association-independent) unless you need the legacy Cantor pairing for id
compatibility. Component order and association
are part of the Rust type and of the `ProductSector` label; `U(1) ⊠ fZ2` and
`fZ2 ⊠ U(1)` are different providers, never automatically equivalent.

## Where an external provider is still second-class

Honest list, at the commit that introduced this document:

- **Cold layout enumeration.** The built-ins additionally implement a sealed
  trait (`LoweredMultiplicityFreeAlgebra`, `tenet-core/src/core_rule_bridge.rs`)
  that decodes a sector once at the miss boundary instead of round-tripping
  `SectorId` per channel. You cannot implement it. Your provider takes the same
  semantic path and pays a constant factor there.
- **Complex coefficients through the typed facade.** A provider with
  `Scalar = Complex64` is expert-layer only until the typed coefficient lane is
  parameterized.
- **Duplicated algebra.** Steps 1 and 2 overlap, as described above.

Each of these is tracked as an issue; none of them changes what your provider
must *mean*.

## Validating your provider

The engine assumes your algebra is consistent and does not verify it. Prove it
against an independent reference, not against TeNeT:

- copy the F/R/dim/twist values from the reference implementation verbatim, and
  say in a comment where they came from — `fibonacci.rs:10-17` and `:189-208`
  are the pattern, including the case where the reference has no override and
  the generic formula applies;
- pin exact values where the reference is exact (`fibonacci.rs:218-221` compares
  bit patterns, not tolerances);
- cover the boundary: an unrepresentable id must produce
  `FusionAlgebraError::InvalidSector` from every checked entry point;
- cover the unit laws if you claim `CanonicalUnitFusionRule`, and the associator
  gauge if you claim `has_trivial_associator_gauge`;
- then run one end-to-end tensor identity — a permute round trip and a
  contraction — because a wrong `fusion_channels` order or a wrong style claim
  survives every per-symbol test.
