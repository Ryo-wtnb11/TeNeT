# Sector ID and storage compatibility

`SectorId` is an opaque, rule-local identifier. Its numeric `usize` value is
not a semantic irrep label, serialization format, or cross-version API.
Persist semantic labels such as `(charge, parity)` or
`(parity, charge, twice_spin)` instead.

## Product-sector codecs

The canonical `ProductFusionRule` is generic over its codec and defaults to
`TensorKitProductCodec`. `PackedProductCodec` remains available when an
application explicitly chooses a fixed-width layout. A provider differing only
in its codec labels the same category with different numeric IDs and a
different `RuleIdentity` — which is exactly why those IDs are not a wire format.

The migration preserves:

- sector labels and degeneracies;
- fusion, duality, F, R, and pivotal semantics;
- tensor results when inputs are matched by semantic fusion-tree labels.

The migration can change:

- numeric `SectorId::id()` values;
- the order returned by `GradedSpace::sectors()`;
- block and reduced-data storage order;
- serialized raw IDs or raw storage buffers;
- the tensor produced by `TensorMap::rand_with_seed` for the same seed.

`rand_with_seed` fills internal storage order. It is reproducible within a
fixed TeNeT version and layout, not a cross-version semantic fixture. Use
`TensorMap::from_block_fn` and inspect provider-labelled fusion trees plus local indices when an input
must remain identical across codec or layout changes.

`GradedSpace::sectors()` is likewise ordered by current internal IDs. Consumers
that need stable behavior must key by the provider's semantic sector type or
apply their own ordering rather than relying on the returned `Vec` order.

## Cache migration

Codec types participate in fusion-rule identity, so in-process TeNeT caches do
not reuse Cantor plans as packed plans. Complete tree-transform plans and
structures are execution-context-local; source-column recoupling rows are
compile-local and are not retained across plan misses. TeNeT no longer reads or writes the former
`tree_transform_plans_v1.bin` and `tree_transform_plans_v2.bin` files. Existing
files are ignored and may be deleted manually. Raw-ID-keyed application caches
must likewise be rebuilt from semantic labels.

This does not change `tenet-network::{save_plan_cache, load_plan_cache}`.
Network contraction-order persistence remains an explicit application opt-in
and stores topology-derived optimizer output, not lowered tree-transform
execution plans.

## Explicit codec choice

`TensorKitProductCodec` is the default codec type parameter of
`ProductFusionRule`. Code that requires a packed numeric layout can name
`PackedProductCodec` explicitly. Neither numeric representation is a semantic
wire format.

Do not mix sectors, spaces, plans, or caches produced by the two codecs. Their
rule identities are distinct even when decoded semantic labels agree.

## Target width

The packed layouts reserve:

- 33 bits for `U(1) x fZ2`;
- 41 bits for `fZ2 x U(1) x SU(2)`.

A 64-bit target is therefore required for these packed product spaces and
for representing the complete encoded `i32` U(1) label set. On a narrower
target the checked codec and space constructors return a width error rather
than truncating, wrapping, or overlapping component bits.

Representation does not imply that the finite `i32` charge window is closed
under U(1) algebra. In particular, `i32::MIN` has no representable negation,
and charge addition can overflow near either endpoint. Checked typed handling
for those algebra boundaries is tracked separately in
[issue #274](https://github.com/Ryo-wtnb11/TeNeT/issues/274); this packed-codec
migration does not wrap, saturate, or reinterpret overflowing charges.
