# Complex device `qr_compact` — leaf M1 (#1271)

Device evidence for admitting `Complex64` and `Complex32` to device
`qr_compact` (bound `CudaFactorizationPayload`) and deleting the
`CudaScalar::DEVICE_CONSTANT_KERNELS` gate and the `CudaQrPayload` marker.
No dependency change (Tenferro 0.6.0, pinned by #1373).

No timing claim is made: `dev` profile, correctness and counter contracts only.

## Environment

| | |
|---|---|
| TeNeT revision | branch `m1-complex-device-qr`: `e3c753e1` (rebased onto `origin/main` `2627f5eb`) plus the review follow-up commit that adds this record; the full five-crate run below is of the pre-rebase `a9053d22` (base `e67b0ada`) |
| Host | `qg1`, sources `/data2/ryo-w/gpu-phase/src` (Cargo.lock deleted and resolved fresh, every `.rs` touched after the sync), shared target `/data2/ryo-w/gpu-phase/gl2-target` |
| GPU | NVIDIA A100-SXM4-40GB, driver 560.35.05, `CUDA_VISIBLE_DEVICES=0`; no compute app of any user on any of the eight GPUs before the runs |
| CUDA / cuTENSOR | 12.6 (V12.6.85) / `libcutensor.so.2.5.0` via `TENFERRO_CUTENSOR_PATH` |
| Toolchain | `rustc 1.96.0 (ac68faa20 2026-05-25)` |
| Build | `dev`, `--no-default-features --features cuda,cpu-faer`, `--test-threads=1` |
| Logs | `/data2/ryo-w/gpu-phase/m1-run.log` (full run), `/data2/ryo-w/gpu-phase/m1b-run.log` (follow-up run) |
| `/data2` | 53% before, 56% after |

## Test counts

Full run (`m0-run.sh` crates and skip list: tenet-operations, tenet-dense,
tenet-tensors, tenet-rs, tenet-network), against the M0 baseline:

| lane | M0 | M1 | change |
|---|---|---|---|
| `--ignored` | 227 | 229 | +3 new device QR tests (rank-deficient/dual multileg laws, hand complex phase, complex QR cost), −1 deleted `complex_device_qr_is_rejected_before_any_device_work` |
| non-ignored | 1670 | 1671 | +1 `complete_structure_warm_hits` (#1374), +2 tenet-dense lib tests from #1373, −1 adapter constant test, −1 marker/constant membership test |
| doc | 98 | 94 | −3 `compile_fail` complex/marker-gate doctests, −1 `f32_cuda_qr` example (the positive twin is `device_qr<D: CudaFactorizationPayload>`) |

All passed, 0 failed.

Follow-up run (review P2-1/P2-2: truly rank-one fixture at every payload, c64
QR at unaligned offsets):

| target | result |
|---|---|
| `typed_cuda_single_precision_factorizations -- --ignored` | 12 passed |
| `typed_cuda_transfer -- --ignored` | 20 passed |
| `cuda_single_precision_probe -- --ignored lu_and_solve` | 2 passed |
| `tenet-rs --doc` | 88 passed, 1 ignored |

## Warm cost

`complex_device_qr_costs_the_real_calls_and_twice_the_bytes`, one warm
`qr_compact` of the U(1) `[2, 3, 2]` fixture, deltas of `cuda_transfer_stats`:

| pair | h2d calls | h2d bytes | d2h calls | d2h bytes | allocs | copies | GEMMs | solver calls |
|---|---|---|---|---|---|---|---|---|
| c64 | 2 | 544 | 0 | 0 | 8 | 6 | 4 | 3 |
| f64 | 2 | 272 | 0 | 0 | 8 | 6 | 4 | 3 |
| c32 | 2 | 272 | 0 | 0 | 8 | 6 | 4 | 3 |
| f32 | 2 | 136 | 0 | 0 | 8 | 6 | 4 | 3 |

Same calls in both fields, twice the uploaded bytes, no download. The two
uploads are the zero output templates (leaf M3 scope).

## LU row-swap probe (raw Tenferro, TeNeT exposes no device `lu`)

`ROW_SWAP` rows `[0, 8+i/4, 0]`, `[2, 1, 0]`, `[4, 0, 1-i/2]` (zero leading
pivot). The device `P` equals the hand partial-pivoting permutation
`P = [[0,0,1],[1,0,0],[0,1,0]]` exactly (asserted entrywise), a non-symmetric
3-cycle, so `P A = L U` is distinguished from `A = P L U`.

| dtype | fixture | `‖P A − L U‖` max | bound | solve residual |
|---|---|---|---|---|
| f32 | dominant (`P = I`) | 0 | 3.433e-5 | 0 |
| f32 | row swap | 0 | 4.578e-5 | 0 |
| c32 | dominant (`P = I`) | 2.980e-8 | 3.433e-5 | 5.960e-8 |
| c32 | row swap | 0 | 4.580e-5 | 1.333e-7 |
