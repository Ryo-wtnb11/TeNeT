# Complete hom-space structure cache — working-set census (#1365, phase 1)

Date 2026-09-24. Host: Apple M4 Max, macOS 15.5, rustc 1.96.0 / cargo 1.96.0,
lldb-1700.0.9.502. TeNeT revision `origin/main` `e354d10d`, with only this
commit's measurement sources added. **No production code or cache bound was
changed.** Cargo lock sha256 `c103cfe3d105561c6316f0a505f78442456a14e84867fdfa95b4410a91d23abf`,
default features (`cpu-faer`), `Runtime::builder().dense_threads(1)`.

Measurement only. This note ends with a recommendation. It is not implemented here.

## Structural findings confirmed at `e354d10d`

All in `tenet-core/src/fusion_space.rs`:

- The lookup is `self.entries.peek(key)` (`:1342`, called from
  `complete_hom_space_structure_cached` under a **read** lock, `:1508`). A hit
  never promotes the entry, and eviction is `pop_lru` (`:1397`). The policy that
  runs is therefore FIFO by admission order, as the doc comment at `:1279` says,
  even though the type is `lru::LruCache`.
- `MAX_ENTRY_BYTES` 1_650_641 is 93.6 % of `BYTE_BUDGET` 1_764_237 (`:1302–1303`).
- `CAP` is 5 (`:1301`). In this census, 48 of the 360 E1 rows reach 5 entries
  from their own inputs and outputs, even with the cache reset before each row.
  The E1 ledger runs its rows back to back without a reset, so there the cache
  stays full. None of those evictions falls in a warm iteration of an E1 row
  (see below).

## Method

The public `complete_hom_space_structure_cache_info()` counters give no key
identity, so the census has three parts:

1. `tenet-network/examples/complete_structure_census.rs` runs the workloads.
   Before each workload it calls `reset_core_intern_tables()`. It records the
   counter deltas around every public call, one `STEP` row per call. It also
   calls a `#[inline(never)] census_mark(step)` function after each call.
2. `benchmarks/complete_structure_census_lldb.py` runs a debug build of that
   example under lldb. It never modifies the process. It breaks on
   `tenet_core::complete_hom_space_structure_cached`, on
   `CompleteHomSpaceStructureCache::admit_built` (for `charged_bytes`) and on
   `census_mark`. It names each key by its HomSpace content: for every
   codomain and domain leg, the duality, the sector ids and the degeneracies,
   read from memory and hashed. The rule is omitted. Each workload uses one
   rule and starts from a reset cache, so the analysis scopes keys by workload.
3. `benchmarks/complete_structure_census.py` checks the trace and replays it.

Validation, both exact:

- Traced lookups equal the observed `hits + misses` on **all 6762 steps**.
- An exact model of today's cache (FIFO, cap 5, the byte budget,
  `max_entry_bytes` bypass) replayed over the trace reproduces the observed
  hits, misses, evictions and bypasses on **all 6762 steps (0 mismatches)**.
  Every re-admission of a key was charged the same bytes as its first admission.
- The release and debug builds produce identical `STEP` rows, so the counters
  do not depend on the build profile.

Replays at other caps use the same trace. "Warm" means E1/conj iterations 2–4,
sweeps 3–6 and iTEBD iterations 11–30. "Compulsory" counts the warm misses on
keys first seen in the warm phase, which no cache can hit. "Min cap" is the
smallest cap at which only those compulsory misses remain. "Live set" is the
largest number of distinct keys touched in one warm iteration. The main replay
bounds entries only. A second pass also applies the current byte budget and
gives identical numbers: bytes never bind in any workload.

Reproduce:

```text
cargo build -p tenet-network --example complete_structure_census
CENSUS_TRACE_OUT=trace.txt lldb -b \
  -o 'command script import benchmarks/complete_structure_census_lldb.py' \
  -o run -o quit target/debug/examples/complete_structure_census > run.log
grep '^STEP' run.log | tr -d '\r' > steps.csv
python3 benchmarks/complete_structure_census.py steps.csv trace.txt
```

The raw data is committed next to this note:
`complete-structure-census-2026-09-24-{steps.csv,trace.txt}.gz`. The script
reads the `.gz` files directly.

## Workloads

| group | what | rows |
|---|---|---|
| `e1` | The E1 ledger inputs (`eager_overhead_ledger.rs`), unchanged: 5 cases × U(1), fZ2×U(1), SU(2) × f64, c64 × 12 ops. Each row runs setup and then 4 calls of its op. | 360 |
| `conj` | #1368: `lazy.contract(m)` / `lazy.compose(a)` with `lazy = a.adjoint()`, compared with the same call on an owned tensor of the adjoint space. f64, all 5 cases × 3 symmetries. | 60 |
| `mps` | Open chain with sites `[vL, p] ← [vR]`, L = 6 and 24, over U(1) (p = ±1) and SU(2) (p = ½). Six sweeps. Each sweep runs a two-site pass (`contract`, `svd_trunc(rank 16 ∧ rtol 1e-10)`, `compose(s, vh)`, `permute`) and then a `qr_compact` + `contract` canonicalization pass, both left to right. | 4 |
| `itebd` | The `tests/itebd_smoke.rs` U(1) iTEBD loop, 30 iterations of two bond updates and one energy evaluation. It runs six-operand `tensor!` networks, `svd_trunc`, `pinv`, `scale` and a `conj` energy network, at χ = 8. | 1 |

## 1. Working set per workload

### E1 rows

Taking the maximum over the 5 cases and both dtypes, the U(1), fZ2×U(1) and
SU(2) figures are **identical**, so one row covers all three symmetries.

| op | lookups / call | distinct keys / call | live set | observed warm misses @ cap 5 | min cap FIFO / LRU |
|---|---:|---:|---:|---:|---:|
| compose | 1 | 1 | 1 | 0 | 1 / 1 |
| contract | 4 | 2 | 2 | 0 | 2 / 2 |
| permute | 1 | 1 | 1 | 0 | 1 / 1 |
| repartition | 1 | 1 | 1 | 0 | 1 / 1 |
| qr_compact | 2 | 2 | 2 | 0 | 2 / 2 |
| svd_compact | 3 | 3 | 3 | 0 | 3 / 3 |
| restrict_leg | 1 | 1 | 1 | 0 | 1 / 1 |
| scale, add, norm, add_adjoint | 0 | 0 | 0 | 0 | — |
| adjoint_data | 2 | 2 | 2 | 0 | 2 / 2 |

In each case (op, symmetry, case, dtype), a single eager op loop touches at most 3
complete structures. None of the 360 rows has a warm miss or a warm eviction
at cap 5. The 18 rows that evict at all do so during setup or on the cold call.

### #1368 conjugated sources

| variant | lookups / call | distinct / call | live set | warm misses @ 5 | min cap FIFO / LRU |
|---|---:|---:|---:|---:|---:|
| contract_lazy | 3 | 2 | 2 | 0 | 2 / 2 |
| contract_owned | 4 | 3 | 3 | 0 | 3 / 3 |
| compose_lazy | 1 | 1 | 1 | 0 | 1 / 1 |
| compose_owned | 1 | 1 | 1 | 0 | 1 / 1 |

These figures are the same for all three symmetries. A lazy source adds no
pressure on this cache and in fact makes one fewer lookup. The per-call rebuild
that #1368 describes (`adjoint_block_structure_view`) re-interns through
`BlockStructure::from_blocks_with_rank` and does not reach this cache. If
#1368 option 2 routes adjoint structures here, each conjugated operand adds at
most one key per call.

### Sweeps and networks: warm miss rate by cap, offline replay

The rates below are warm misses divided by warm lookups, with the entry cap as
the only bound.

| workload | warm calls | warm lookups | compulsory | distinct / call | live set | min cap FIFO / LRU | FIFO 5 | FIFO 8 | FIFO 16 | FIFO 32 | FIFO 64 | LRU 5 | LRU 8 | LRU 16 | LRU 32 | LRU 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mps/U1/L6 | 120 | 268 | 2 | 5 | 21 | 25 / 24 | 56.7% | 53.7% | 31.3% | 1.1% | 0.7% | 55.2% | 53.7% | 26.9% | 0.7% | 0.7% |
| mps/SU2/L6 | 120 | 260 | 1 | 3 | 19 | 23 / 22 | 55.4% | 50.8% | 29.2% | 1.2% | 0.4% | 53.8% | 50.8% | 21.5% | 0.4% | 0.4% |
| mps/U1/L24 | 552 | 1348 | 3 | 5 | 35 | 54 / 36 | 59.3% | 44.5% | 16.0% | 5.2% | 0.2% | 54.3% | 44.5% | 15.1% | 6.5% | 0.2% |
| mps/SU2/L24 | 552 | 1332 | 17 | 5 | 53 | 85 / 54 | 59.2% | 47.5% | 24.2% | 15.5% | 2.3% | 55.2% | 47.5% | 23.9% | 16.3% | 1.3% |
| itebd/U1 | 320 | 900 | 0 | 7 | 27 | 27 / 27 | 77.8% | 77.8% | 61.1% | 0.0% | 0.0% | 77.8% | 77.8% | 57.8% | 0.0% | 0.0% |

The observed counters at today's cap 5 match the model exactly:

| workload | warm misses / warm lookups | evictions |
|---|---:|---:|
| mps/U1/L6 | 152 / 268 | 239 |
| mps/SU2/L6 | 144 / 260 | 227 |
| mps/U1/L24 | 800 / 1348 | 1247 |
| mps/SU2/L24 | 788 / 1332 | 1231 |
| itebd/U1 | 700 / 900 | 1062 |

**FIFO versus LRU.**

- **E1 and #1368 rows:** the two policies give identical results. Every live
  set is at most 3.
- **iTEBD:** the network touches 27 keys in a cycle, and the two policies are
  identical at every cap except 16, where they differ by 3 points (61.1 % vs
  57.8 %). Both reach zero at cap 27.
- **Sweeps:** LRU needs a smaller cap to reach the compulsory floor. For L = 24
  it needs 36 instead of 54 (U(1)) and 54 instead of 85 (SU(2)). A sweep's
  bulk sites reuse the same bond spaces, and FIFO evicts those hot keys by
  admission age. At every cap in the 5–64 table, though, the miss rates of the
  two policies differ by at most 8 points, and at U(1) L = 24, cap 32, LRU is
  the worse of the two (6.5 % vs 5.2 %). Below the live set both miss about
  half the time, and above it both miss only compulsorily.

So **the cap decides the outcome, and the policy changes it at the margin
only**. Switching `peek` to `get` at cap 5 would leave the sweeps and the
network at 54–78 % warm misses.

**Scaling.** The live set of a sweep grows with chain length, but more slowly
than linearly: 21 → 35 keys for U(1) and 19 → 53 for SU(2) from L = 6 to 24.
Bulk bonds repeat, and the ends and the not-yet-converged spectra add distinct
keys. A per-call distinct count of at most 7 does not predict the loop's live
set.

## 2. Entry bytes

Per workload, the distinct keys' `charged_bytes`, which is exactly what the
cache charges:

| group | distinct keys | min | median | p90 | max | max live set | max live-set bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| e1/U1 | 330 | 3532 | 8456 | 13268 | 22756 | 3 | 29912 |
| e1/fZ2xU1 | 330 | 3660 | 8584 | 13396 | 22884 | 3 | 30296 |
| e1/SU2 | 330 | 3773 | 10545 | 24165 | 47101 | 3 | 51492 |
| conj/U1 | 59 | 3532 | 8456 | 21972 | 22756 | 3 | 31304 |
| conj/fZ2xU1 | 59 | 3660 | 8584 | 22100 | 22884 | 3 | 31688 |
| conj/SU2 | 59 | 3773 | 10545 | 45101 | 47101 | 3 | 53339 |
| mps/U1/L6 | 49 | 3532 | 4923 | 8533 | 12192 | 21 | 113131 |
| mps/SU2/L6 | 45 | 3155 | 4455 | 6281 | 7429 | 19 | 77664 |
| mps/U1/L24 | 59 | 3532 | 4923 | 8533 | 12192 | 35 | 207785 |
| mps/SU2/L24 | 90 | 3155 | 4455 | 6281 | 7429 | 53 | 244748 |
| itebd/U1 | 73 | 2576 | 7263 | 14670 | 24078 | 27 | 301046 |

For the `e1` and `conj` groups, "distinct keys" counts per-row distinct keys
summed over rows. For those groups, "max live set" and "max live-set bytes" are
per row.

Measured entries range from 2.6 KB to 47 KB. The largest measured live set
costs 301 KB, which is 17 % of today's byte budget. The largest entry is 2.9 %
of today's `max_entry_bytes`. Bytes never bound any measured workload; the
entry count did.

## 3. Reference: a comparison point, not evidence

TensorKit 0.17.1 (`~/.julia/packages/TensorKit/DQFb5`):

- `src/spaces/structure.jl`: `sectorstructure` (`:41`, `CacheStyle` `:74`) and
  `degeneracystructure` (`:114`, `:198`) are `@cached` with `GlobalLRUCache`.
- `src/auxiliary/caches.jl:19,162`: that cache is
  `LRU{Any,Any}(; maxsize = DEFAULT_GLOBALCACHE_SIZE[])`, with a default of
  10^4 entries. It is bounded by entry count, not bytes, and it is a true LRU.
- LRUCache.jl 1.6.2 (`src/LRUCache.jl:141–143`): `get!` takes the cache's
  single lock on every lookup, including hits, to move the entry to the front.

TensorKit therefore pays an exclusive lock on every hit, and it caps entries
about 200× above anything measured here. That size suits Julia, where
structure objects are small and GC-managed. It is not a target for TeNeT. What
transfers is the order of magnitude: a structure cache has to hold a whole
sweep or network, not the handful of keys of one call.

QSpace has no corresponding cache. It reconstructs structural data per
operation.

## Recommendation (not implemented; for the supervisor)

1. **Bytes bind; the entry cap is only a backstop.** Set
   `BYTE_BUDGET = 4 MiB` (4_194_304). That is 13.9× the largest measured live
   set (301 KB). It also still holds 54 entries of the largest measured size
   (47 KB × 54 = 2.5 MB), which is the LRU requirement of the heaviest sweep.
   Set `CAP = 1024`. At the measured median entry (4–11 KB), 1024 entries are
   4–11 MB, above the budget, so in practice bytes are the binding limit. The
   cap only stops the entry count from growing without bound when entries are
   very small, and it stays 10× below TensorKit's 10^4.
2. **Shrink `MAX_ENTRY_BYTES` to budget / 16 = 256 KiB** (262_144). That is
   5.6× the largest measured entry (47 KB), so an outlier is rejected rather
   than allowed to evict everything else and occupy the budget alone, which is
   possible at today's 93.6 %.
3. **Keep FIFO for now, document it, and leave the hit path on the read
   lock.** At the recommended bounds, every measured live set fits many times
   over, and FIFO and LRU give the same misses (the compulsory ones only).
   Switching to `get` needs `&mut self`, so every hit would take the write lock
   (`:1508` today takes a read lock). That serializes the hot path and works
   against #1366; TensorKit accepts that cost, and TeNeT need not. If eviction
   quality under pressure later matters (the sweep evidence: LRU reaches its
   floor at about ⅔ of the cap FIFO needs), use CLOCK / second-chance: set a
   per-entry `AtomicBool` "referenced" on a hit under the read lock, and give
   referenced entries a second pass at eviction under the write lock. That is
   close to LRU without a write lock on hits. Either way, fix the naming:
   today the cache is documented as FIFO but typed as `lru::LruCache` and
   evicts with `pop_lru`.
4. **Gate for the fix.** Replay this census against the new bounds, and assert
   in a test that the sweep and iTEBD workloads have no non-compulsory warm
   misses and that E1 rows have `evictions == 0` on warm calls. The committed
   trace allows the replay offline, so the bounds can be re-derived without
   re-tracing.

Residual scope: the sweeps use χ ≤ 16 and L ≤ 24, and the network is only
U(1) iTEBD at χ = 8. Entry bytes grow with block count and rank, not with
degeneracy, so larger χ with the same sector content does not change them.
More sectors or higher rank (for example PEPS, or the B3 boundary-MPS shapes,
which are not re-run here) do, and 4 MiB is sized from this census alone. The
SU(2) L = 24 sweep still makes 17 compulsory misses in sweeps 3–6, because its
spectra had not converged.
