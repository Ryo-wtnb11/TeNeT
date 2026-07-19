# SU(N) fusion-symbol table generator (Stage B3b / B3c-1)

Generates the group-agnostic tabulated fusion-symbol blobs loaded by the Rust
`TabulatedFusionRule` provider (`tenet-core/src/su3.rs`):

- `tenet-core/src/su3_table.bin` — the canonical SU(3) `dim ≤ 27` table,
  embedded with `include_bytes!` (`Su3FusionRule` = the alias over it);
- `tenet-core/src/testdata/su4_table.bin` — a small SU(4) `dim ≤ 15` table,
  loaded at runtime with `TabulatedFusionRule::from_bytes` by the B3c-1
  "new group = data only" smoke test.

The N-/F-/R-/dim/dual/Frobenius–Schur values come straight from
[SUNRepresentations](https://github.com/QuantumKitHub/SUNRepresentations.jl)'
implementation of the TensorKitSectors sector interface for `SUNIrrep{N}` — they
are the reference, not a transcription of one. `gen.jl` only serialises them.

## N-parametric

```
julia --project=<env> gen.jl [N] [dim_cut] [out_path]
julia --project=<env> gen.jl 3 27                                     # SU(3), default out
julia --project=<env> gen.jl 4 15 tenet-core/src/testdata/su4_table.bin  # SU(4) smoke
```

where `<env>` is a Julia environment with `SUNRepresentations` + `TensorKitSectors`.
`N` and the `dim` cut are the ONLY group-specific inputs; everything else goes
through the group-agnostic sector interface. A new group is DATA ONLY — no Rust
changes (proven by `b3c1_su4_table_is_data_only`).

`SectorId` is the dense index into the irrep list sorted by `(dim, label…)`, so
the vacuum (all-zero Dynkin label) is id 0 (matches `FusionRule::vacuum`).
Fusion-channel sequences separately preserve SUNRepresentations'
`directproduct` order because TensorKit exposes that order as fusion-tree
inner-line basis order; numeric `SectorId` order is not a substitute.
The production reference is TensorKit revision
`cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91`,
`src/fusiontrees/iterator.jl::_fusiontree_iterate`, where provider product
iteration determines inner-line order, and
`src/fusiontrees/fusiontrees.jl::treeindex_data/treeindex_map`, where that
order becomes an observable basis index. QSpace revision
`dd2cc7e10dc7d3917b23309a44d1fe67adb4dc43` has no corresponding public
fusion-tree iterator/channel-order contract; `Source/QSpace.hh::QSpace` and
`Source/QSpace.cc::QSpace` instead retain provider-specific QIDX/DATA/CGR row
order. Within fusion-tree basis construction, TeNeT therefore preserves the
pinned SUN `directproduct` sequence for inner lines, while coupled block groups
remain numeric-`SectorId` ordered. The Julia generator records the provenance
of the current offline artifact; TeNeT's runtime and Rust build do not require
Julia, and a future Pure Rust generator can preserve the same ordering
contract.

## Bounded table (exact or `Err`, never silently truncated)

A finite `dim` cut does not close under fusion. Pairs whose product escapes the
set (e.g. SU(3) `8⊗10 ∋ 35`) are the provider's hard-error boundary: block
dimensions are either exactly full-`SU(N)` or an `Err`. The **frontier shell**
(integer-only: out-of-table channels, escaping pairs, one-hop returns) lets the
Rust coupled-sector fold (`su3.rs coupled_sector_fold`) classify sectors as
clean / tainted / escaped instead of panicking mid-enumeration. No frontier F/R
symbols are stored — a coupled sector whose trees pass through a frontier inner
line cannot be recoupled with this table, so it is reported as `Err` (Stage B3c
lifts the cut), never silently enumerated as a truncated tree set.

## Row-major

Julia arrays are column-major; a column-major transcription bug bit Stage B2b.
`gen.jl` flattens every F/R block **row-major** on the generator side (matching
the Rust `GenericFArray::get` / `GenericRMatrix::get` indexing). The reader
copies bytes verbatim and must never re-transpose. Integer and floating-point
fields are emitted explicitly in little-endian order, independent of the Julia
host. A trailing FNV-1a-64 of the
payload is stored in the header and re-checked by the loader, so a transpose or
truncation mistake fails loudly at first use.

Before writing, the generator checks every F/R record shape against independent
`Nsymbol` calls, checks R unitarity, and runs TensorKitSectors pentagon and
hexagon equations on the four lowest-dimensional irreps. It then decodes the
emitted bytes independently and compares every R/F key, shape, and row-major
coefficient with the in-memory SUNRepresentations source records. CI also
generates a small SU(3) table whose F records contain asymmetric outer-
multiplicity axes, so the decoder cannot pass by treating every block as a
scalar. The Rust loader then
reconstructs admissible keys from serialized fusion multiplicities and rejects
missing, extra, malformed, non-finite, or non-unitary symbol records.

## Binary format v3 (magic `TFR3`)

Adds a group tag and variable-length labels over v2 (magic `SU3T`): the header
carries `group_n = N`, and each irrep/frontier label is `rank = N-1` Dynkin
components (v2 hard-coded SU(3)'s two, `p:u8, q:u8`). For SU(3) the payload is
byte-identical to v2 (rank 2), so the provenance hash is unchanged; only the
4-byte magic + 4-byte `group_n` header differs. See `gen.jl` for the full layout.

## Provenance

| field | SU(3) blob | SU(4) smoke blob |
|-------|-----------|------------------|
| group / dim cut | SU(3), dim ≤ 27 | SU(4), dim ≤ 15 |
| generated | 2026-07-19 | 2026-07-19 |
| SUNRepresentations | 0.4.0 | 0.4.0 |
| TensorKitSectors | 0.3.9 | 0.3.9 |
| Julia | 1.11.6 | 1.11.6 |
| format version | 3 | 3 |
| irreps / covered pairs / R / F | 17 / 82 / 731 / 76853 | 7 / 17 / 59 / 589 |
| frontier / escaping / one-hop | 41 / 207 / 697 | 15 / 32 / 105 |
| size | 1 874 710 bytes | 13 473 bytes |
| payload FNV-1a-64 | `0x1d8f726bf9928176` | `0xcffdd18bbba9155a` |

> Size note: the full SU(3) `dim ≤ 27` table is 1.87 MB (the 135 805 F-symbol
> `f64`s are the irreducible bulk), stored as a little-endian binary blob loaded
> with `include_bytes!` + a one-pass parser (see `su3.rs`), not emitted as Rust
> source. Then run `cargo test -p tenet-core su3` — the provider's FNV self-check
> and the TK oracle tests re-validate the blob.
