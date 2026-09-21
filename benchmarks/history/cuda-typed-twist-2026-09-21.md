# Device `twist` / `twist_inverse` — evidence record (G2b-t, #1330)

Base: `origin/main` 91d15f97 (single-precision device payloads #1340), rebased
from ab47d16c (device `*_overwrite_into` #1339), on which the device run below
was taken; the rebase touched only the operation matrix, and the typed hunk
merged unchanged. Branch `g2bt-device-twist`. No dependency change.

The body is one generic over `CudaPayload`, so after #1340 it is instantiated
for `f32`/`Complex32` as well. `±1` is exact in single precision, so no new
numerical question arises, but there is no single-precision twist fixture: the
matrix records that column as `NEEDS-PROOF` under
[#1336](https://github.com/Ryo-wtnb11/TeNeT/issues/1336), not as proved.

## What landed

`twist` and `twist_inverse` on `TensorMap<R, D, CudaStorage<D>>`, generic over
`CudaPayload`, sharing one private `twist_with_inverse_cuda`
(`tenet/src/typed.rs`). `flip` stays `UnsupportedOnDevice`.

## Reference record

- TensorKit `twist!`, `src/tensors/indexmanipulations.jl:62-77` @`cfaa073`:
  tests `has_shared_twist(f₁, f₂, i)` per fusion-tree pair and applies
  `scale!(t[f₁, f₂], θ)` to that block. The decomposition — one scalar per
  block, evaluated at call time, no structural tensor and no basis change — is
  what both the Host and this device path implement.
- QSpace: **no ribbon-twist path exists.** QSpace's fermionic/ribbon handling
  is carried in its permutation and contraction sign bookkeeping; there is no
  `twist`-equivalent entry point to port a layout or batching technique from.
  Concrete absence, recorded as the policy requires, not an unexamined one.
- Rust/TeNeT deviation: the factor table is a Host `Vec` built before the
  device lease, and the per-block scaling is one `cuda_region_axpby` per block
  instead of an in-place `scale!`, because Tenferro 0.5.0 has no in-place
  strided scale (`cuda_region_axpby` cannot alias source and destination).

## Host branch -> device branch

| # | Host `twist_with_inverse` (`tenet/src/typed.rs`) | Device `twist_with_inverse_cuda` |
|---|---|---|
| 1 | leg `>= rank` -> `InvalidArgument("{name} leg {leg} out of range for rank {rank}")` | same, same string |
| 2 | `legs.is_empty()` -> `self.clone()` | same |
| 3 | `reject_unbraided_nonunit_legs(provider, hom, legs, name, true)` | same call, same position |
| 4 | lazy adjoint -> `parent.twist_with_inverse(adjoint_axes, !inverse)?.adjoint()` | same, recursing on the device method; `.adjoint()` takes no lease |
| 5 | compact spectrum -> scaled `SectorSpectrum`, or a clone when every factor is 1 | unreachable (`to_cuda` densifies): `Diagonal` payload -> `UnsupportedOnDevice("twist requires dense CUDA storage")` |
| 6 | `twist_is_identity_over_blocks` -> `self.clone()` | same, **before** step 5's storage check, so an all-ones twist clones whatever the payload is |
| 7 | copy the payload, then `scale_blocks_impl` skipping factor-1 blocks | output = #740 zero upload, then one `cuda_region_axpby` per block with that block's factor |

Step 6 ahead of step 5 is the one intentional order change, and it makes the
device *more* Host-faithful: Host's compact arm also returns a clone when every
factor is one.

Steps 1-6 and the whole factor table run before `lease_cuda()`. One lease is
taken, nothing under it leases again.

## How θ reaches the kernel, and why

The per-block factor is `twist_block_factor(provider, key, nout, legs,
inverse)` — the existing Host helper, unchanged, so no new categorical logic
exists on the device path — and it is passed as the `alpha` of
`cuda_region_axpby` with `CudaRegionCoefficient::One`, i.e. as the contraction
descriptor's own scale. Nothing is uploaded for it.

That is the opposite of the structural-coefficient decision for tree transforms
(#1318: coefficients are 1x1 *data* operands), and the reason the two differ is
the value domain:

- the device impl admits `R: MultiplicityFreeRigidSymbols<Scalar = f64>`, so
  `twist_scalar` is an `f64`;
- every such provider returns `±1`: `+1` for a bosonic rule
  (`tenet-sectors/src/abelian.rs`, `su2.rs`, `cu1.rs`), `-1` exactly on the odd
  sector of fermion parity (`abelian.rs:503`), and a product rule multiplies
  its factors (`product_rule.rs:512`). `twist_factor_with_inverse` conjugates,
  which is the identity on a real factor;
- a product over `legs` of `±1` is `±1`, so **θ is never zero**.

A zero would have been fatal twice over: `cuda_region_axpby` rejects a zero
descriptor alpha outright, and a zero alpha lets CUDA skip the source read,
which erases the NaN/Inf propagation the Host has. Since θ is never zero, NaN
and real infinities propagate as on Host, and a warm call uploads nothing but
its output.

Not exactly Host, in one disclosed case: Host skips a factor-1 block with a bit
copy while the device always multiplies, so an infinite **complex** entry comes
back NaN in both components — for θ = -1 Host gives `(-inf, NaN)` where the
device gives `(NaN, NaN)`. This is the #1301 deviation already disclosed at
`tenet-dense/src/cuda_adapter.rs:1279-1285`; `f64` and every finite payload are
exact.

Complex twist factors (Fibonacci is `Scalar = Complex64`) are excluded at
compile time by the impl bound, not at runtime; if a `Scalar = f64` provider
ever returned a non-`±1` value the descriptor scale would still be correct, and
only a zero would be a boundary. The value domain is pinned **without a device**
by `tenet/tests/typed_transform_host_side.rs::a_fermionic_twist_only_ever_keeps_or_negates_an_entry`,
which runs in ordinary CI for fZ2, fZ2 x U(1) and fZ2 (x) SU(2).

## Cost contract

Per call, warm:

- 1 H2D of `required_len * size_of::<D>()` bytes (the #740 zero initialisation
  of the output), 0 D2H, 1 device allocation;
- one kernel submission per non-empty block, no coefficient buffer;
- nothing prepared, cached or retained: a twist compiles no
  `TreeTransformStructure`, so `cuda_tree_transform_stats()` is unchanged by
  it.

Measured on the fZ2 `[leg, leg] <- [leg, leg]` fixture (41 elements, 8 blocks,
f64): first call `h2d_calls = 2, h2d_bytes = 336, device_allocs = 2,
gemm_calls = 8` — 328 output bytes plus the context's shared one-element `1`
operand, created once per dtype per context; every later call
`h2d_calls = 1, h2d_bytes = 328, d2h = 0/0, device_allocs = 1, gemm_calls = 8`.
`twist_inverse` has the identical profile.

Residuals, disclosed rather than fixed here:

1. a bitwise whole-buffer device copy followed by an in-place per-block sign
   scale would submit `1 + |{blocks with θ = -1}|` kernels instead of one per
   block, and would preserve padding bytes; Tenferro 0.5.0 has no in-place
   strided scale, so it is not expressible today;
2. the #740 zero upload is still the whole output, as for every other returning
   device operation;
3. flat elements belonging to no block are zero here where Host copies them
   through. Only reachable under a padded expert layout; the same convention
   the device structural transforms have had since #1322;
4. the device NoBraiding preflight has no test of its own. No production
   provider is `NoBraiding`, so one needs a custom provider, and a device
   tensor over a custom provider needs both the `cuda` feature and a real GPU —
   an ungated test cannot reach the device call site. The call is the same
   statement on the same shared helper as Host, whose rejection *is* gated
   ungated by `tenet/tests/typed_facade.rs` (planar Z2 fixture, `:8919`);
5. the zero-length upload path (`required_len == 0` with blocks present, i.e.
   a zero extent) is not reached by any fixture: a space with no coupled sector
   has no block and short-circuits first.

## Correctness evidence

Device (`--ignored`, A100, `tenet/tests/typed_cuda_twist.rs`): device == Host
for fZ2 (dual and non-dual legs, codomain and domain, single-axis, multi-axis,
repeated legs), fZ2 x U(1) and fZ2 (x) SU(2), `f64` and `Complex64`, on lazy
adjoint operands for both directions, plus `twist ∘ twist_inverse == id`
exactly, the bosonic clone, the empty leg list, a space with no coupled sector
(zero blocks, so a clone — the zero-length upload path is not reached by it)
and the out-of-range rejection with the Host's own message. Every
fermionic fixture asserts non-vacuity through `assert_signs_only`: at least one
entry is negated, and no entry is scaled by anything but `±1`.

Contracts (`tenet/tests/typed_cuda_transform_contracts.rs`): the warm transfer
and allocation profile above, the short circuits at zero counters, and the
rejections at zero counters with `cuda_tree_transform_stats()` unchanged.

An all-legs fermionic twist is the identity in value (the factors multiply to
the block's even total parity) but is **not** a short circuit, because the
detection tests each leg's own factor rather than the product — Host publishes
a fresh unscaled copy there, and the device does the same per-block work rather
than inventing a cheaper answer. Both halves are pinned.

## Device run (A100, `cuda,cpu-faer`, `--test-threads=1`)

qg1, CUDA 12.6, a GPU with no other process, private target
`/data2/ryo-w/gpu-phase/g2bt-target` (removed afterwards). Verbatim:

```
tests/typed_cuda_twist.rs -- --ignored
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.26s

tests/typed_cuda_twist.rs (non-ignored)
test result: ok. 0 passed; 0 failed; 5 ignored; 0 measured; 0 filtered out; finished in 0.00s

tests/typed_cuda_transform_contracts.rs -- --ignored
test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 4.13s

tests/typed_transform_host_side.rs (ungated, no device)
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.13s

cargo test -p tenet-rs --doc --no-default-features --features cuda,cpu-faer
test result: ok. 82 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 9.14s
```

Full workspace suites, same build:

- `--lib --tests -- --ignored` (skips as the runbook prescribes), 117 test
  binaries: **171 passed, 0 failed**;
- `--lib --tests` without `--ignored`: **2532 passed, 0 failed, 177 ignored**.

Local gates (macOS, private target, removed afterwards): `cargo fmt --all
--check`; `cargo clippy --workspace --all-targets -- -D warnings` and the same
with `--no-default-features --features cuda,cpu-faer`; `cargo test -p tenet-rs
-p tenet-network`; doctests in both feature sets (68 and 82 passed);
`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` and the `cuda`
doc build of `tenet-rs`.
