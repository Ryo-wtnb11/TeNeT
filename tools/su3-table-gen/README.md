# SU(3) fusion-symbol table generator (Stage B3b)

Generates `tenet-core/src/su3_table.bin`, the checked-in table loaded by the
Rust `Su3FusionRule` provider (`tenet-core/src/su3.rs`) via `include_bytes!`.

The N-/F-/R-/dim/dual/Frobenius–Schur values come straight from
[SUNRepresentations](https://github.com/QuantumKitHub/SUNRepresentations.jl)'
implementation of the TensorKitSectors sector interface for `SUNIrrep{3}` — they
are the reference, not a transcription of one. `gen.jl` only serialises them.

## Irrep set

Every `SUNIrrep{3}` with `dim ≤ 27` — 17 irreps:
`1, 3, 3̄, 6, 6̄, 8, 10, 10̄, 15, 15̄, 15′, 15̄′, 21, 21̄, 24, 24̄, 27`.
This cut closes `8⊗8 = 1+8+8+10+10̄+27` (the SU(3) adjoint / Heisenberg
motivation). Pairs whose fusion escapes the set (e.g. `8⊗10 ∋ 35`) are the
provider's hard-error boundary: `fusion_channels` panics, it never truncates.

`SectorId` is the dense index into the irrep list sorted by `(dim, p, q)`, so
vacuum `(0,0)` is id 0 (matches `FusionRule::vacuum`).

## Row-major

Julia arrays are column-major; a column-major transcription bug bit Stage B2b.
`gen.jl` therefore flattens every F/R block **row-major** on the generator side
(matching the Rust `GenericFArray::get` / `GenericRMatrix::get` indexing). The
Rust reader copies bytes verbatim and must never re-transpose. A trailing
FNV-1a-64 of the payload is stored in the header and re-checked by the loader,
so a transpose or truncation mistake fails loudly at first use.

## Regenerate

```
julia --project=<sunenv> tools/su3-table-gen/gen.jl
```

where `<sunenv>` is a Julia environment with `SUNRepresentations` 0.4.0 and
`TensorKitSectors`. Then run `cargo test -p tenet-core su3` — the provider's
FNV self-check and the TK oracle tests re-validate the new blob.

## Provenance (current blob)

| field | value |
|-------|-------|
| generated | 2026-07-11 |
| SUNRepresentations | 0.4.0 (Julia pkg slug `BM32Z`) |
| TensorKitSectors | 0.3.4 |
| Julia | 1.11.6 |
| tenet git commit | 816c35ac4bf735ee0a9799dab912672785590bd5 |
| irreps / covered pairs / R / F | 17 / 82 / 731 / 76853 |
| size | 1 866 475 bytes (1.866 MB) |
| payload FNV-1a-64 | `0xbfd30b91b8a025fd` |

> Size note (Stage B3b STOP condition): the full `dim ≤ 27` table is 1.866 MB,
> above the ~1.5 MB threshold for a checked-in `.rs` constant. It is therefore
> stored as a compact little-endian binary blob and loaded with `include_bytes!`
> + a one-pass parser (see `su3.rs`), not emitted as Rust source. The F-symbol
> coefficients (135 805 `f64`s) are the irreducible bulk; a smaller `dim` cut
> would drop `8⊗8 → 27` and defeat the adjoint physics motivation.
