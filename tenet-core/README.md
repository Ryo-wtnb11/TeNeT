# tenet-core

Expert structural layer below the `tenet` facade: spaces, fusion trees and
block layouts, with no tensor type of its own. Entry points are
`TensorMapSpace`, `FusionTensorMapSpace`, `SectorLeg`, fusion-tree and block
keys, block views, and storage traits. It validates spaces and layouts but
does not provide high-level execution.

Enable `racah-generated` to pass generated SUN support through to sectors.
