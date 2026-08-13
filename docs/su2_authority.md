# Historical SU(2) authority migration record

> **Historical revision-specific evidence.** The benchmark and compatibility
> sweep below describe the recorded migration pair, not current-main behavior.
> Snapshot authority: TeNeT
> `c612b4ff5c0b9285fa6ef8617b5c5472cba3287e`. The current dependency authority
> is the versioned `racah` requirement in `tenet-sectors/Cargo.toml`; see the
> [historical-record index](history.md).

At the recorded migration revision, `tenet-sectors` delegated base/default-feature
SU(2) representation algebra, F/R/Frobenius--Schur coefficients, and coefficient
caches to `racah` revision
`489f367e7d0dce85842f15b253be0b74c4c47340`. The compatibility baseline is
TeNeT commit `6a3bbe365982928c8e8ce014f19e13ac15a14002`.

The SU(2) `RuleIdentity` contains racah's authority fingerprint, the
`SectorId.id() == 2j` encoding, maximum `2j = 254`, and ascending step-two
channel order. Racah's process-global 3j, 6j, and derived-F tiers have a
documented aggregate ceiling of 192 MiB. Their public controls are reexported
as `tenet_sectors::su2_coefficient_cache`; TeNeT core and operation-cache
resets do not clear them, and racah resets do not clear TeNeT structural caches.

## Compatibility Protocol

Build the baseline and candidate, then compare only their public scalar APIs:
`f_symbol_scalar`, `r_symbol_scalar`, `frobenius_schur_phase_scalar`,
`fusion_channels`, and the checked fusion/N-symbol APIs.

1. Run six nested loops `dj1` through `dj6`, each `0..=12`, and compare every
   `f_symbol_scalar(dj1, dj2, dj3, dj4, dj5, dj6)`. Also compare the boundary
   tuples `(d, d, 0, 0, 0, d)` for `d = 32, 64, 127, 128, 253, 254`.
2. Map each finite, non-NaN `f64` bit pattern to ordered ULP space with
   `if sign { !bits } else { bits | (1 << 63) }`, and require distance at most
   two. Treat `+0.0` and `-0.0` as equivalent; otherwise require matching
   finite/nonfinite classification and sign.
3. Enumerate `dj1`, `dj2`, and `dj3` over `0..=254`, retaining cases with
   `dj1 + dj2 <= 254`, and compare `r_symbol_scalar(dj1, dj2, dj3)` exactly.
   Enumerate `dj = 0..=254` and compare every Frobenius--Schur phase exactly.
4. Check `fusion_channels` is `|dj1-dj2|, |dj1-dj2|+2, ..., dj1+dj2` in that
   order whenever the sum is representable. Check checked fusion validates
   left then right then closure, while checked N-symbol validates coupled then
   left then right then closure.
5. Replay boundary failures exactly. `SU2Irrep::from_twice_spin(255)` panics
   `SU(2) doubled spin exceeds the supported maximum 254`, while
   `from_sector_id(SectorId::new(255))` panics
   `SU(2) sector exceeds the supported maximum doubled spin 254`. Checked
   fusion returns `InvalidSector { sector: 255 }` for `(255, 254)`,
   `(254, 255)`, and `(255, 256)`, proving left-before-right validation; it
   returns `FusionNotRepresentable { left: 128, right: 127 }` for `(128, 127)`.
   Checked N-symbol returns `InvalidSector { sector: 255 }` for `(254, 255, 0)`
   and `InvalidSector { sector: 256 }` for `(255, 0, 256)`, proving the coupled
   label is first. `r_symbol_scalar(1, 1, 255)` is `0.0`; and
   `r_symbol_scalar(254, 1, 253)` panics with
   `SU(2) fusion closure exceeds the supported maximum doubled spin 254`.

The adapter rejects labels above 254 in constructors and checked APIs. The
historical infallible R surface deliberately differs: after validating left,
right, and closure, a nonchannel coupled label, including one above 254,
returns `0.0`. In-range inadmissible F/R values also remain `0.0`.

The recorded pre-switch run covered 4,826,815 F cases (maximum two ULP, or
`2.220446049250313e-16` absolute; no sign, zero/nonzero, or nonfinite
mismatch), 8,323,200 R triples, and all 255 FS labels. The F sweep retained
1,254,500 racah cache bytes, below 192 MiB.

## Post-switch Benchmark

On the same machine in release mode with `RAYON_NUM_THREADS=1`,
`microbench_fusion` took three warmed samples per revision and degeneracy.
The exact-SU(2) rows were:

| `d` | operation | candidate (us) | baseline (us) | candidate/baseline |
| --- | --- | ---: | ---: | ---: |
| 4 | compose | 29.43 | 29.29 | 1.0048 |
| 4 | swap | 40.45 | 39.92 | 1.0133 |
| 4 | swap+out | 47.96 | 47.67 | 1.0061 |
| 16 | compose | 100885.66 | 100581.30 | 1.0030 |
| 16 | swap | 103279.56 | 103227.00 | 1.0005 |
| 16 | swap+out | 105961.82 | 104999.77 | 1.0092 |

Checksums match exactly: at `d = 4`, compose/swap/swap+out are
`193.5`, `259.375`, and `8.375`; at `d = 16`, they are `0.125`,
`-147.125`, and `-53.125`. The maximum regression is 1.33%, within the 3%
gate. The benchmark emits no allocation metric.

`operation_matrix.sh` was not run: its fixed rows are facade
permute/decomposition operations and cannot select the relevant SU(2) path.
`microbench_fusion` directly exercises the existing coefficient-authority path.
