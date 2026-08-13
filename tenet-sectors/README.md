# tenet-sectors

Provider vocabulary for TeNeT's symmetry and category layers. Expert users
implement or select `FusionRule`, `SectorCodec`, `CheckedFusionAlgebra`, and
the capability traits; built-in providers include U(1), Z/N, fZ2, SU(2), and
Fibonacci. `ProductFusionRuleExt` builds ordered, nested products: factor
order and association remain part of the provider and sector type.

The `racah-generated` feature enables the generated SUN provider. It does not
turn arbitrary product labels or raw `SectorId` values into stable wire data.
