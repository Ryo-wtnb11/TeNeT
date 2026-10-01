# Dense LQ causal experiments

These diagnostic runs are excluded from the final comparison. Full raw CSVs
and disassembly were archived locally at `/tmp/issue-1676-experiments-4746bbd0`;
only targeted values and executable hashes are kept here. All cases below use
dual U1, four sectors, degeneracy 64, dense storage, and compact LQ. Values
are per-process medians in nanoseconds under the final report’s settings.

## Original regression

Baseline 34aef370 versus pre-fix candidate ffa882ea, five alternating pairs:

| Dtype | Baseline runs 1–5 | Pre-fix runs 1–5 |
| --- | --- | --- |
| f32 | 201625, 201522, 200222, 199347, 199327 | 208760, 207387, 210135, 206718, 210518 |
| f64 | 264210, 264064, 263660, 263647, 262427 | 270052, 269879, 271835, 268308, 271258 |

## Isolated variants

Each variant starts from ffa882ea and changes only the described helper or
branch. Two pairs alternate pre-fix/variant then variant/pre-fix. Scoped
Release package cleaning precedes each build. These are causal diagnostics,
not standalone speed claims; early process-first cases showed occasional noise.

- **Ablation:** remove only diagonal admission in compact LQ. Regression persists.
- **Boundary:** force the existing adjoint helper out of line. Regression persists.
- **Indexed:** nested `Vec::push`, with reserve and traversal unchanged. Slower;
  discarded.
- **Range:** exact-size index range passed to `Vec::extend`. Restores real
  timing without observed complex-control harm; adopted in 122ef8c8 and then
  independently rebuilt for the final five-pair comparison.

| Variant | Dtype | Pre-fix runs 1, 2 | Variant runs 1, 2 |
| --- | --- | --- | --- |
| ablation | f32 | 210485, 210210 | 209460, 210587 |
| ablation | f64 | 271781, 270260 | 269750, 270854 |
| ablation | c32 | 426989, 426975 | 427066, 429750 |
| ablation | c64 | 635377, 634785 | 634266, 636552 |
| boundary | f32 | 210270, 211227 | 210629, 210450 |
| boundary | f64 | 273343, 272712 | 272545, 269960 |
| boundary | c32 | 431302, 432350 | 430072, 431002 |
| boundary | c64 | 638229, 640460 | 636947, 642395 |
| indexed | f32 | 210950, 210320 | 223504, 224443 |
| indexed | f64 | 272197, 272487 | 286935, 289268 |
| indexed | c32 | 430877, 432060 | 455252, 452420 |
| indexed | c64 | 639408, 640625 | 670150, 665981 |
| range | f32 | 209908, 209835 | 200635, 199085 |
| range | f64 | 272372, 270856 | 261216, 261552 |
| range | c32 | 429912, 431060 | 428047, 427102 |
| range | c64 | 637700, 653645 | 634929, 632662 |

`nm` / `llvm-objdump` showed scalar load loops in the real baseline adjoint
append helper and an out-of-line StepBy `try_fold` call per row in the
pre-fix/forced-boundary builds. The range retains one reserve, append order,
column-major traversal and conjugation. Dispatch ablation and the failed
push variant distinguish this code-generation effect from admission cost.

## Saved executable SHA256

```text
baseline / pre-fix
13e4f99efc16e6c3077e53ee92295ddf96288ec5333ff73aed422587d5293f67  1676-baseline-bench
3bb658f025de866f78e01bcf789caf7e4e7e1a27d50f9c3311a3ef1c9d162dcd  1676-candidate-bench
variants
b17d2314937684f449c885352f8c128a3e5363faa80e9849e2d560fb985e45ef  1676-ablation-bench
85595867a6f9e1d66768a11b55f0e8def8c4e239be4042f09078900f83b7dbaa  1676-boundary-bench
68316ce8601b12d630398a5cc2f34e68064bfd51351b46ea6931846498e6fd83  1676-indexed-bench
965bab0e2eea04251b74425f3aab7fa472666f94c4e13af26a8f478ad225fad0  1676-range-bench
```

## Invalid early run

An earlier shared-target build reused one executable for both revisions
(both SHA256 started `68e1f6`). Those CSVs are excluded and retained only in
`/tmp/1676-excluded-stale`. Cargo freshness across worktrees was insufficient.
Accepted builds use scoped `cargo clean --release -p tenet-matrixalgebra -p
tenet-rs`, distinct saved executable hashes, and the semantic storage probe.
No final speed estimate uses the invalid run.
