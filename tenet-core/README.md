# tenet-core

Core tensor maps, fusion-tree keys, block layouts, sector vocabulary, and
storage traits. This crate contains data and validation, not high-level
execution. Its fusion-tree model is informed by TensorKit and QSpace, while
TeNeT's Rust types and validation rules are authoritative.

`ProductSector` is a sector label for a product fusion provider. It is not the
leg-level `ProductSpace` concept from TensorKit; TeNeT represents tensor legs
as `SectorLeg` values collected in `FusionProductSpace`.
