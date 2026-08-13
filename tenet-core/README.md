# tenet-core

Expert structural layer for the static, low-level `TensorMap<T, NOUT, NIN>`;
it is distinct from the dynamic user facade in `tenet`. Entry points are
`TensorMapSpace`, `FusionTensorMapSpace`, `SectorLeg`, fusion-tree and block
keys, and storage traits. It validates spaces and layouts but does not provide
high-level execution.

Enable `racah-generated` to pass generated SUN support through to sectors.
