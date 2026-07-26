# Provenance and references

Every number this crate ships is traceable to the rows below. The crate itself
performs no derivation of the *source* data: it parses the pinned text, shifts
one-based source labels to zero-based `SectorId`s, and derives only the
API-level quantities that TensorKitSectors also derives (dimension,
Frobenius–Schur phase, twist, A/B).

## Layer 1 — exact algebraic source

| Field | Value |
| --- | --- |
| Repository | `JCBridgeman/smallRankUnitaryFusionData` |
| Commit | `a6ba1c308ddcecbb03b9651a25fa0d1afcfb7546` (2023-03-06) |
| Content | Multiplicity-free unitary fusion categories of rank ≤ 6, retained as exact Mathematica expressions (`Sqrt[...]`, `Root[...]`, roots of unity) |
| Ring indexing | AnyonWiki, "List of small multiplicity-free fusion rings" |
| Export program | `exportData.nb` at that commit |

## Layer 2 — numerical export (the projection this crate consumes)

`exportData.nb` evaluates each exact expression and writes decimal text. The
notebook was read at the pinned commit; the two operative calls are

```
Q = N[<exact F/R expression>, 25]
ToString[DecimalForm[Re[Q], {Infinity, 20}]]   (* and likewise Im *)
```

so the shipped text carries 25-significant-digit evaluations rendered with
exactly 20 decimal places. **The decimal files are therefore already a
numerical projection of the exact source, not the exact source itself.**

## Layer 3 — distribution

| Field | Value |
| --- | --- |
| Repository | `lkdvos/CategoryData.jl` |
| Package version | `0.3.6` |
| Package commit | `e793a08a093f6ba890a32c7e57d8e8b347441058` |
| Artifact name | `fusiondata` |
| Artifact tag | `data-v0.1.3` |
| Artifact git-tree-sha1 | `90a414867a392b7b811ba6c8092e5e1a848dc4c4` |
| Tarball URL | `https://github.com/lkdvos/CategoryData.jl/archive/refs/tags/data-v0.1.3.tar.gz` |
| Tarball SHA-256 | `586f48411e0c3aa8780959d2395724bf6d1de4e4bcf785eb84864e0d5c045cf6` |

The tarball was downloaded once, its SHA-256 checked against
`CategoryData.jl/Artifacts.toml` at the pinned package commit, and three files
copied byte-for-byte into `data/categorydata-v0.1.3/` under their upstream
paths. Their SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| `Nsymbols/FR_2_1_0_2.txt` | `a7a6d179ab51c62229d1c8d3aeb0c0bfc44f811dec033525d6c53ebf2cb31290` |
| `Fsymbols/FR_2_1_0_2_0.txt` | `00cbae541c159fa4eb5f3142c3759f626966cbac001cf0eb585b76d2657cb30e` |
| `Rsymbols/FR_2_1_0_2_0_0.txt` | `24a653d14322cd5f177e577b7a1328e4ed92217a0cbc7c613d5be0eeef032b2d` |

## Category identity, ordering, and gauge

| Field | Value |
| --- | --- |
| CategoryData alias | `Fib` |
| Type | `PMFC{2, 1, 0, 2, 0, 0}` (rank 2, multiplicity 1, selfduality 0, ring index 2, category index 0, braid index 0) |
| Object ordering | Source file order, one-based: `1 = 𝟙`, `2 = τ`. Frozen; not re-sorted |
| Braiding choice | Braid index `0`, i.e. `Rsymbols/FR_2_1_0_2_0_0.txt`. Index `1` is the mirror braiding and is a *different* category for identity purposes |
| Gauge | Source gauge, loaded verbatim. The only transformation applied is the one-based → zero-based label shift |

`Fib = PMFC{2,1,0,2,0,0}` is fixed by `CategoryData.jl/src/aliases.jl` at the
pinned package commit; the artifact file names are built by
`src/artifacts.jl::{N,F,R}_artifact` from the type parameters.

Source record layouts (`CategoryData.jl` data-branch `README.md`, identical to
the upstream `smallRankUnitaryFusionData` README):

```
Nsymbols: a b c Nabc
Fsymbols: a b c d α e β μ f ν Re Im
Rsymbols: a b c α μ Re Im
```

## Layer 4 — the `Complex64` projection epoch

Epoch identifier: **`tenet-category-data/projection/2026-07-27`**.

The epoch fixes, once, how a decimal record becomes a `Complex64`:

1. Split a record on ASCII whitespace; the trailing two fields are the real and
   imaginary decimal literals.
2. Convert each literal with Rust's `str::parse::<f64>()`, i.e. the
   correctly-rounded IEEE-754 binary64 conversion under round-to-nearest,
   ties-to-even. Rust and Julia both guarantee correct rounding here, so the
   crate reproduces `CategoryData.jl`'s `parse(Float64, …)` bit-for-bit.
3. Pair them as `Complex64 { re, im }`. No re-normalisation, no gauge fixing,
   no rounding to "nice" values, no promotion of a real table to `f64` storage.
4. Unlisted `N`/`F`/`R` records are exact zeros.

Changing any step is a new epoch identifier, and therefore a new
`RuleIdentity`.

## Derived quantities

Derived from the pinned tables using the TensorKitSectors 0.3.9 formulas
(`~/.julia/packages/TensorKitSectors/VuA9Z/src/sectors.jl`):

| Quantity | Formula | TKS location |
| --- | --- | --- |
| `dim(a)` | `abs(1 / F(a, ā, a, a, 1, 1))` | `dim_from_Fsymbol`, `sectors.jl:431-439` |
| `sqrtdim` / `invsqrtdim` | `sqrt(dim)` / `inv(sqrt(dim))` | `sectors.jl:440-441` |
| `κ(a)` | `sign(F(a, ā, a, a, 1, 1))` | `frobenius_schur_phase_from_Fsymbol`, `sectors.jl:463-469` |
| `θ(a)` | `Σ_{b ∈ a⊗a} dim(b)/dim(a) · R(a, a, b)`, summed in ascending `b` | `twist_from_Rsymbol`, `sectors.jl:646-647` |
| `A`/`B` | `tenet-sectors` `MultiplicityFreeRigidSymbols` defaults | `sectors.jl:501-511`, `:543-551` |

## Reference fixture

`tests/fixtures/fib-categorydata-v0.1.3.txt` was emitted by
`tests/fixtures/generate_fixture.jl` run against a Julia 1.11.6 project holding
`CategoryData v0.3.6` (pinned rev above) and `TensorKitSectors v0.3.9`. It is a
genuine independent oracle: the values come out of CategoryData.jl's own
parser, sparse-array indexing, and dual/dimension/twist derivation, not out of
a re-print of the source files.

Regenerating it requires Julia and network access and is therefore an offline,
manual step. Nothing in the shipped crate invokes Julia, Mathematica, or the
network.
