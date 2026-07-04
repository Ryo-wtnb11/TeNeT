# TensorKit 互換対応表(命名改定の正本)

作成日: 2026-07-04。`tensorkit_semantic_naming_audit.md` の実施決定版。
「現行名 → 改定名」はこの表が正本。改名は tier ごとに 1 コミット、372 テスト緑維持。

## 層の定義(監査メモどおり)

- **ユーザー層**(将来の高レベル API): TensorKit 語彙(`permute`, `tsvd`, …)。本表の後に別途設計。
- **expert 層**: TensorOperations の `tensorcontract!` 系に対応する `*_into` 関数群。
- **内部実行層**: plan / cache / workspace / backend。数学名を混ぜない。

## P0: contraction spec 語彙

| TensorKit / TO.jl | 現行 TeNeT | 改定名 | 備考 |
|---|---|---|---|
| `pA/pB/pAB/conjA/conjB` の束 | `TensorContractAxisSpec` | `TensorContractSpec` | doc に pA/pB/pAB 対応を明記 |
| `pAB` 省略(既定出力順) | `TensorContractAxisSpec::canonical(l, r)` | `TensorContractSpec::with_default_output_order(l, r)` | 「canonical」廃止 |
| `pAB` | `AxisPermutation` | `OutputAxisOrder { Default, Axes }` | contraction 出力順専用 |
| (Rust 補助) | `OwnedTensorContractAxisSpec` | `TensorContractSpecOwned`(internal 寄せ) | cache key 用補助型 |

## P1: 操作名 / 実行計画名

| TensorKit | 現行 TeNeT | 改定名 | 備考 |
|---|---|---|---|
| `permute!` / `braid!` / `transpose!` | `tree_pair_transform_into*` | `permute_into` / `braid_into` / `transpose_into`(+ `_with_context` 変種) | 操作は `TreeTransformOperation` で内部共有。`TreePairTransform*` は内部 replay 名として存続 |
| (内部) | `TreeTransformOperationKey` | `TreeTransformOperation`(public 操作)/ cache key は internal `TreeTransformCacheKey` | |
| (実装制約) | `all_codomain_tree_transform_into_with_context` | `pub(crate)` へ降格 | permute/braid/transpose facade の内部 |
| `contract!` 内部の core 形 | `canonical_dst` / `canonical_axes` / `canonical_dst_nout/nin` | `core_dst` / `core_axes` / `core_dst_open_lhs_rank` / `core_dst_open_rhs_rank` | 「canonical」廃止、open/contracted 語彙へ |
| 同上 | `lhs_canonical_nout/nin`, `rhs_canonical_nout/nin` | `lhs_open_rank` / `lhs_contract_rank` / `rhs_contract_rank` / `rhs_open_rank` | |
| 同上 | `CanonicalFusionBlockContractPlan` | `FusionBlockContractPlan` | pack/scatter 撤去後は唯一の形なので修飾不要 |
| 同上 | `is_canonical_fusion_block_contract` | `is_core_form_fusion_block_contract` | |
| `tensorcontract!` の prepared 相当 | `tensorcontract_fusion_explicit_plan_into*` | expert 層は `prepare_tensorcontract_fusion` + `execute_prepared_tensorcontract_fusion`(T18 で導入済み)に一本化。`explicit_plan_into*` は deprecated → 削除 | 型 `TensorContractFusionExplicitPlan` は internal `FusionContractPlan` |
| (内部ルート名) | `dynamic_plan` / `DynamicTree` | internal のみ許容(public docs から排除) | |

## P2: constructor / core 名(doc 優先、改名は最小)

| 現行 | 対応 | 備考 |
|---|---|---|
| `from_degeneracy_shapes` | 名前維持 + doc に「fusion-tree **subblock** ごとの degeneracy shape」を明記 | TensorKit の block(coupled)/subblock(tree)区別を明示 |
| `from_vec_with_fusion_space` | 名前維持(低レイヤ constructor)| 高レベル constructor はユーザー層で提供 |
| `coefficients_src_by_dst` | `recoupling_coefficients_dst_src` | `U[dst, src]` の線形写像向きを名前に |
| `fusion_tree_keys_from_external_sectors` | 名前維持 + doc に domain 側 dualize を明記 | |
| `tensorcontract_homspace` | 名前維持 + doc で `tensorcontract!` lowering 相当と明記 | `compose` との関係を doc に |
| `BlockStructure::packed_column_major` | tutorial から排除(tests / low-level docs 限定)| |

## expert 層の TensorKit 対応(維持する名前)

```text
TensorKit tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β)
TeNeT     tensorcontract_into(&mut C, A, B, spec, α, β)
TensorKit tensoradd!(C, A, pA, conjA, α, β)
TeNeT     tensoradd_into(&mut C, A, ..., α, β)
TensorKit tensortrace!(C, A, p, q, conjA, α, β)
TeNeT     tensortrace_into(&mut C, A, ..., α, β)
```

`TensorMap` / `TensorMapSpace` / `FusionTreeHomSpace` / `FusionProductSpace` /
`SectorLeg` / `DenseExecutor` / `DenseKernelBackend` / `DenseView` は維持。

## 実施順

1. P0(spec 語彙)— 1 コミット
2. P1(操作名 / plan 名)— 1〜2 コミット
3. P2(doc + 2 件の改名)— 1 コミット
4. その後、ユーザー層 API 設計(TensorKit 準拠 / Runtime 内蔵 / メソッド先行)
