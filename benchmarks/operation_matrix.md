# Operation matrix

`operation_matrix.sh` is a read-only smoke benchmark for the public user API.
It fixes Rayon to one worker and reports a fresh-process (cold) sample followed
by repeated-process (warm) samples. It intentionally measures the existing
owned-returning API; no allocator or cache toggles are enabled.

The destination rows are not silently substituted with owned calls. TensorKit
benchmarks `permute!` and `svd_compact!` into caller-owned storage. TeNeT's
typed destination primitive is `tenet_tensors::permute_into` (and
`transpose_into`); constructing a destination requires the corresponding
`FusionTensorMapSpace`, so those rows remain an explicit follow-up rather than
an incomparable number.

Run with:

```sh
REPS=5 RAYON_NUM_THREADS=1 benchmarks/operation_matrix.sh
```

This harness is intentionally outside CI: it is diagnostic evidence, not a
correctness gate. Semantic coverage remains in `tenet/tests/user_api.rs` and
`tenet/tests/user_decomp.rs`.
