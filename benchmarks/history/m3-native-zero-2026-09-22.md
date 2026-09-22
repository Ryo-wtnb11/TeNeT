# M3 (#740): device zero initialisation A/B, 2026-09-22

This note is an observation, not a pass/fail gate. It compares:

- **base** `ed2aa606`: `origin/main`, which creates a fresh buffer by uploading a host `Vec` of zeros;
- **branch** `7c496030`: `m3-native-zero-init`, which calls `CudaDenseStorage::zeros`. That is Tenferro `alloc_zero_output`: bind, `get_resource`, flush, then `cuMemsetD8Async`, all inside `with_cubecl`, which flushes on entry and on exit.

## Setup

- **Machine:** qg1, A100-SXM4-40GB (GPU 0, no other process on it), driver 560.35.05, CUDA 12.6, cuTENSOR 2.5.0, rustc 1.96.0.
- **Stack and settings:** tenferro 0.6.0 and t4a-cubecl 0.10.1. Both revisions used the same `Cargo.lock`. Every thread knob was set to 1, and the builds were release.
- **Build isolation:** each revision was built from its own source tree into its own target directory. Both were deleted afterwards.
- **Run order:** base, then branch, then base, then branch. Each cell below is the median of the two passes' medians.
- **Raw data:** the per-pass CSVs, `compare.py` and the sweep source are in `m3-native-zero-2026-09-22/`.

## Small-output sweep (`m3_zero_sweep.rs`)

- **Operations:** warm `compose` and `contract` over one U(1) sector with a small inner extent (`k = 4`), so output initialisation is a large share of each call.
- **Samples:** 51 per row, after 5 warm-ups.
- **`submit`** times the call alone. TeNeT has no public device sync.
- **`e2e`** times the call followed by a 1-element `to_host`. That readback drains the single CubeCL stream.
- **`sync`** is that readback on its own (about 27–35 µs).

| dtype | op | output | submit base µs | submit branch µs | ratio | e2e base µs | e2e branch µs | ratio |
|---|---|---|---|---|---|---|---|---|
| f64 | compose | 64 B | 63.5 | 102.4 | 1.61 | 100.1 | 135.6 | 1.35 |
| f64 | compose | 4 KiB | 69.5 | 106.4 | 1.53 | 107.1 | 139.9 | 1.31 |
| f64 | compose | 256 KiB | 178.6 | 112.8 | 0.63 | 215.9 | 146.5 | 0.68 |
| f64 | compose | 16 MiB | 27291 | 112.0 | 0.004 | 27377 | 146.6 | 0.005 |
| f64 | contract | 64 B | 63.4 | 102.3 | 1.61 | 100.0 | 135.2 | 1.35 |
| f64 | contract | 4 KiB | 70.5 | 105.0 | 1.49 | 107.7 | 138.2 | 1.28 |
| f64 | contract | 256 KiB | 178.7 | 112.1 | 0.63 | 215.7 | 145.7 | 0.68 |
| f64 | contract | 16 MiB | 27263 | 110.9 | 0.004 | 27356 | 145.6 | 0.005 |
| c64 | compose | 64 B | 72.1 | 97.1 | 1.35 | 112.5 | 128.3 | 1.14 |
| c64 | compose | 4 KiB | 76.1 | 96.7 | 1.27 | 116.5 | 127.9 | 1.10 |
| c64 | compose | 256 KiB | 182.2 | 105.3 | 0.58 | 220.4 | 137.2 | 0.62 |
| c64 | compose | 16 MiB | 32191 | 102.6 | 0.003 | 32277 | 143.5 | 0.004 |
| c64 | contract | 64 B | 72.3 | 96.8 | 1.34 | 112.3 | 128.2 | 1.14 |
| c64 | contract | 4 KiB | 74.9 | 101.5 | 1.35 | 114.0 | 133.7 | 1.17 |
| c64 | contract | 256 KiB | 182.4 | 104.7 | 0.57 | 220.2 | 136.4 | 0.62 |
| c64 | contract | 16 MiB | 32432 | 102.2 | 0.003 | 32518 | 143.1 | 0.004 |

How to read it:

- **Branch is flat in size.** It costs about 97–113 µs to submit at every size. That is a fixed initialisation cost, as expected for a memset preceded by blocking CubeCL round trips.
- **Base grows with the size.** It pays for a host `Vec`, its H2D and the staging, from about 63 µs at 64 B up to 27–32 ms at 16 MiB.
- **Where the curves cross.** It lies between 4 KiB and 256 KiB. At or below 4 KiB the branch is 20–40 µs slower per returned output; at 256 KiB it is about 1.6× faster; at 16 MiB it is about 250× faster.
- **Base is noisy between passes.** It varied across passes: pass 1 submitted 64 B in 55 µs, pass 2 in 89 µs. The branch agreed within 1–2 µs. Both passes are listed in `compare.py`'s output.

## `cuda_operation_matrix` (default sizes, f64 and c64, warm cuda rows)

- The matrix times submission only, with its `norm()` barrier outside the timer.
- Ratio is branch/base per row, then the median or geometric mean over rows.
- The byte columns sum over the rows.

| dtype | operation | rows | median ratio | geomean | rows slower | H2D bytes base → branch | host alloc bytes base → branch |
|---|---|---|---|---|---|---|---|
| f64 | all | 160 | 1.038 | 1.119 | 146/160 | 11611328 → 2419904 | 90244464 → 53513712 |
| c64 | all | 152 | 1.018 | 1.053 | 99/152 | 21286656 → 4838144 | 125439712 → 59677280 |
| f64 | compose / contract_direct / permute / chain3 | 8 each | 1.02–1.04 | 1.07–1.12 | 8/8 | output H2D → 0 | about −80% |
| c64 | compose / contract_direct / permute / chain3 | 8 each | 0.98–1.01 | 0.99–1.01 | 3–5/8 | output H2D → 0 | about −80% |
| f64 | scale / add_owned / add_lazy_fold | 8 each | 1.36–1.57 | 1.30–1.46 | 8/8 | about 480 KB → 64–128 B | about −97% |
| f64/c64 | norm / inner (tiny partials) | 8 each | 1.28 | 1.21 | 8/8 | 1.7–3.3 KB → 0 | ≈ |
| f64/c64 | svd / eigh / qr | 8 each | 1.00–1.01 | 1.00–1.03 | mixed | factor-output uploads gone (spectrum uploads stay) | −8 to −50% |

The matrix fixtures are small: under 64 KiB per output at the default sizes. They therefore sit near or below the crossover, which is why cheap operations (`scale`, `add`, `norm`, `inner`) show the fixed cost of about 40–55 µs most clearly.

## Structural cause and what this does not decide

- **Cost of the new path.** The branch replaces an O(bytes) host allocation plus H2D with a size-independent device memset. In Tenferro's route that memset sits behind about four blocking CubeCL server round trips: the `with_cubecl` entry flush, `get_resource`, the flush before the raw enqueue, and the exit flush. Those round trips, not the memset, are the fixed cost above.
- **What would lower it.** Choosing between the two paths by output size would be a size-threshold dispatch, which the policy does not allow. Removing the fixed cost needs fewer round trips on the Tenferro side, such as a `with_cubecl`-free zero allocation or a single flush.

That choice belongs to the supervisor or maintainer.
