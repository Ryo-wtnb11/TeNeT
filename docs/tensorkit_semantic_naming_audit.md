# TensorKit 互換意味論のための命名監査メモ

作成日: 2026-07-03

> **実施済み (2026-07-04)**: P0 / P1 の改名は完了。現行名との対応は
> [`tensorkit_compatibility_table.md`](tensorkit_compatibility_table.md) が正本。
> 本メモ中の「現在の名前」は改名前のものを指す。

## 目的

TeNeT の API 名を、TensorKit / TensorOperations.jl の意味論に照らして見直す。

問題は単に「低レイヤ名が public facade に漏れている」ことだけではない。同じアルゴリズム、同じデータフローを実装していても、名前が TensorKit の概念を正しく表していなければ、利用者にも実装者にも誤った理解を誘導する。

したがって、命名は次の順で判断する。

1. TensorKit / TensorOperations.jl の意味論と同じか。
2. 同じなら、その名前がその意味論を正確に表しているか。
3. public API に出してよい抽象度か。
4. Rust の所有権、cache、workspace、backend 境界の都合が、数学的 API 名に混ざっていないか。

## 基本方針

TeNeT では名前を 3 層に分ける。

```text
ユーザー層:
  einsum!, tensorcontract!, tensoradd!, tensortrace!, permute!, braid!, transpose!

TensorKit expert 層:
  tensorcontract_into(dst, A, pA, conjA, B, pB, conjB, pAB, alpha, beta)
  tensoradd_into(dst, A, pA, conjA, alpha, beta)
  tensortrace_into(dst, A, p, q, conjA, alpha, beta)

内部実行層:
  PreparedContract
  ContractCorePlan
  FusionBlockReplayPlan
  TreeTransformer
  cache/workspace/backend-specific replay objects
```

TensorKit の `@tensor A[i; j]` は、通常の scalar indexing ではなく、マクロが読む index notation である。マクロは `pA`, `pB`, `pAB`, `conjA`, `conjB` を作り、`tensorcontract!` / `tensoradd!` / `tensortrace!` に lower する。

TeNeT でも同じ構造に寄せる。低レイヤは `pA/pB/pAB` 相当を正確に持ち、ユーザー層ではそれを隠す。

## P0: すぐに見直すべき名前

### `TensorContractAxisSpec::canonical`

該当:

- `tenet-tensors/src/axis.rs`

現在の意味:

```rust
TensorContractAxisSpec::canonical(lhs_contracting_axes, rhs_contracting_axes)
```

これは「canonical form」ではない。意味は以下。

```text
lhs の指定軸と rhs の指定軸を縮約する。
出力軸は lhs の open axes を元の順に並べ、その後 rhs の open axes を元の順に並べる。
```

TensorKit 対応:

```text
pA  = (lhs open axes, lhs contracted axes)
pB  = (rhs contracted axes, rhs open axes)
pAB = identity/default output order
```

問題:

- `canonical` という名前は categorical canonical form、canonical gauge、canonical tree basis を連想させる。
- TensorKit の public API にこの概念名は出てこない。
- 実際には `pAB` を省略した default output order でしかない。

方針:

- public tutorial からは隠す。
- 低レイヤに残す場合でも `canonical` は避ける。
- 候補:
  - `TensorContractSpec::from_contracted_axes_default_output(...)`
  - `TensorContractSpec::lhs_open_then_rhs_open(...)`
  - `TensorContractSpec::with_default_output_order(...)`
  - TensorKit expert 層では `pAB_identity` / `Index2Tuple` 相当として表す。

### `AxisPermutation`

該当:

- `tenet-tensors/src/axis.rs`

現在の意味:

```text
contract 後の open axes を destination の axes にどう並べるか。
```

問題:

- 一般の tensor axis permutation に見える。
- contraction 文脈では `pAB` / output index order のことであり、source tensor の `permute` とは違う。
- TensorKit の `permute(t, p::Index2Tuple)` と名前が衝突しやすい。

方針:

- contraction 文脈では `OutputIndexOrder` または `OutputAxisOrder` の方がよい。
- TensorKit expert 層では `pAB` に対応する名前を明示する。

候補:

```rust
enum OutputIndexOrder<'a> {
    Default,
    Axes(&'a [usize]),
}
```

### `TensorContractAxisSpec`

該当:

- `tenet-tensors/src/axis.rs`

現在の中身:

```text
lhs_contracting_axes
rhs_contracting_axes
output_permutation
lhs_conjugate
rhs_conjugate
```

問題:

- `AxisSpec` だと縮約軸だけの指定に見える。
- 実際には TensorKit の `pA`, `pB`, `pAB`, `conjA`, `conjB` をまとめている。
- output order と conjugation を持つなら、名前は contract 全体の index lowering を表すべき。

方針:

- TensorKit expert 層では `Index2Tuple` 相当を導入するか、`TensorContractIndexSpec` に寄せる。
- ユーザー層では macro / builder で隠す。

候補:

```rust
TensorContractSpec
TensorContractIndexSpec
ContractIndexOrder
```

より TensorKit に寄せるなら:

```rust
Index2Tuple { codomain: Vec<usize>, domain: Vec<usize> }

TensorContractSpec {
    lhs: Index2Tuple,
    lhs_conj: bool,
    rhs: Index2Tuple,
    rhs_conj: bool,
    output: Index2Tuple,
}
```

### `OwnedTensorContractAxisSpec`

該当:

- `tenet-tensors/src/axis.rs`

問題:

- `Owned` は Rust 実装都合であり、数学的意味論ではない。
- public に見える名前としては弱い。

方針:

- 原則 internal。
- public に必要なら `TensorContractSpecOwned` のように補助型であることを明示する。

## P1: TensorKit の操作名ではなく TeNeT の実装名になっているもの

### `tree_pair_transform_*`

該当:

- `tenet-tensors/src/lib.rs`
- `tenet-tensors/src/facade.rs`

現在の意味:

```text
fusion-tree pair basis transform を replay する。
```

TensorKit 対応:

```text
permute / permute!
braid / braid!
transpose / transpose!
add_permute!
add_braid!
add_transpose!
GenericTreeTransformer / AbelianTreeTransformer
```

問題:

- `tree_pair_transform` は実装構造名であり、数学操作名ではない。
- TensorKit ユーザーが期待するのは `permute`, `braid`, `transpose`。
- 「tree pair」という語は必要な内部データ構造を説明しているだけで、API 操作の意味を表していない。

方針:

- public 操作は TensorKit に合わせる。
- 内部 replay / cache 名としてのみ `TreePairTransform` を残す。

候補:

```rust
permute_into(...)
braid_into(...)
transpose_into(...)
add_permute_into(...)
add_braid_into(...)
add_transpose_into(...)
```

### `TreeTransformOperationKey`

該当:

- `tenet-operations/src/transform_key.rs`

問題:

- `OperationKey` は cache key に見える。
- public 操作名としては `Key` が不要。
- `Permute` variant が braiding-aware categorical permutation なのか、単なる dense axis permutation なのか名前だけでは分からない。

方針:

- public 操作として出すなら `TreeTransformOperation`。
- cache key なら internal にして `TreeTransformCacheKey`。
- TensorKit 操作に対応する public 関数名は `permute`, `braid`, `transpose`。

### `all_codomain_tree_transform_into_with_context`

該当:

- `tenet-tensors/src/facade.rs`

問題:

- 実装制約名であり、TensorKit の操作名ではない。
- public に出すと「all-codomain という別の数学操作」があるように見える。

方針:

- internal helper に落とす。
- public には `permute_into` / `braid_into` / `transpose_into` の一部として見せる。

## P1: 実行計画名が意味論名に混ざっているもの

### `tensorcontract_fusion_explicit_plan*`

該当:

- `tenet-tensors/src/lib.rs`
- `tenet-tensors/src/contract/fusion/plan.rs`

問題:

- `explicit` が何に対して explicit なのか分からない。
- TensorKit の user-facing / expert-facing API には出ない実行計画名。
- 実行計画が必要なことは Rust/HPC では正しいが、それを数学操作名の前面に出すべきではない。

方針:

- public user 層では隠す。
- expert 層なら `prepare_contract` / `compile_contract`。
- internal 型なら `FusionContractPlan` または `PreparedFusionContract`。

### `canonical_dst`, `canonical_axes`, `canonical_dst_nout`, `canonical_dst_nin`

該当:

- `tenet-tensors/src/contract/fusion/plan.rs`
- `tenet-tensors/src/contract/dynamic_space.rs`
- `tenet-tensors/src/contract/dynamic.rs`

現在の意味:

```text
lhs transformed to [open, contracted]
rhs transformed to [contracted, open]
その中間 contraction result の destination
```

問題:

- `canonical` が categorical canonical form を連想させる。
- 実際には block GEMM 用の core/intermediate contraction space。
- TensorKit の `contract!` 内部で BLAS 可能な形へ揃える話であり、public 概念ではない。

候補:

```text
contract_core_dst
matmul_core_dst
intermediate_contract_dst
core_contract_axes
```

方針:

- public API からは消す。
- internal 名としても `canonical` は避ける。

### `lhs_canonical_nout/nin`, `rhs_canonical_nout/nin`

該当:

- `tenet-tensors/src/contract/fusion/plan.rs`

現在の意味:

```text
lhs_canonical_nout = lhs open axes count
lhs_canonical_nin  = lhs contracted axes count
rhs_canonical_nout = rhs contracted axes count
rhs_canonical_nin  = rhs open axes count
```

問題:

- `nout/nin` は TensorMap の codomain/domain rank を連想させる。
- しかし contraction core では `lhs_canonical_nin` や `rhs_canonical_nout` は「contracted axes 側」を表している。
- TensorKit / TensorOperations の `pA=(openA, contractA)`, `pB=(contractB, openB)` の語彙で見れば自然だが、`canonical_nout/nin` と呼ぶと TensorMap の domain/codomain 意味と混ざる。

候補:

```text
lhs_open_rank
lhs_contract_rank
rhs_contract_rank
rhs_open_rank
core_dst_open_lhs_rank
core_dst_open_rhs_rank
```

方針:

- 内部でも `canonical_nout/nin` は避ける。
- contraction core の用語は `open` / `contracted` で統一する。

### `dynamic_plan`

問題:

- `dynamic` は Rust 実装の cache / allocation 方針に見える。
- TensorKit の意味論上は「動的な別演算」があるわけではない。

方針:

- internal 実行ルート名としてのみ許容。
- public に必要なら `prepared` / `cached` / `workspace` の文脈に限定する。

## P2: 低レイヤ名としては正しいが、ユーザー層には重いもの

### `from_degeneracy_shapes`

該当:

- `tenet-core/src/lib.rs`

意味:

```text
FusionTreeHomSpace の各 fusion-tree block に dense degeneracy dimensions を与える。
```

これは意味論としては正しい。ただし、TensorKit ユーザーが最初に触る constructor としては低レイヤすぎる。

方針:

- core constructor として残してよい。
- 上位 facade では `TensorMapSpace` / `HomSpace` / `TensorMap` 風の builder を用意する。

追加注意:

- TensorKit では `block` と `subblock` が明確に分かれている。
- TeNeT の `degeneracy_shapes` は、TensorKit の coupled-sector matrix block shape ではなく、fusion-tree subblock ごとの dense degeneracy shape を渡している。
- したがって名前だけ見ると、どの粒度の shape なのかが曖昧。

候補:

```text
from_subblock_degeneracy_shapes
from_fusion_tree_subblock_shapes
from_fusion_tree_degeneracy_shapes
```

既存名を残す場合も、doc では「fusion-tree subblock ごとの shape」と明記する。

### `from_vec_with_fusion_space`

該当:

- `tenet-core/src/lib.rs`

意味:

```text
既に構築済みの FusionTensorMapSpace に flat storage を attach する。
```

Rust としては正しいが、TensorKit の `TensorMap(data, codomain, domain)` の使い心地からは遠い。

方針:

- 低レイヤ constructor として残す。
- public tutorial では、より高レベルな constructor を優先する。

### `BlockStructure::packed_column_major`

意味:

```text
block fixture / explicit storage layout constructor
```

方針:

- 通常ユーザー向け tutorial には出さない。
- tests / low-level docs に限定する。

## 追加で監査対象にする名前

### `coefficients_src_by_dst`

該当:

- `tenet-tensors/src/tree_transform/plan.rs`

問題:

- 実体は recoupling matrix の `U[dst, src]`。
- 名前が保存順を説明しているだけで、線形写像としての向きが分かりにくい。
- TensorKit の `fusiontreetransform` / `GenericTreeTransformer` との対応では、source basis から destination basis への recoupling map であることが重要。

候補:

```text
recoupling_matrix_dst_src
recoupling_coefficients_dst_src
tree_transform_matrix_dst_src
```

### `fusion_tree_keys_from_external_sectors`

該当:

- `tenet-core/src/lib.rs`

問題:

- domain 側 external sectors は内部 tree sector として扱うときに dualize される。
- TensorKit でも `subblock(t, sectors)` では domain 側を dualize して fusion-tree key に落とす。
- 名前だけでは domain dualization が見えない。

候補:

```text
fusion_tree_keys_from_external_sectors_with_domain_dual
fusion_tree_keys_from_tensor_indices
subblock_keys_from_external_sectors
```

### `compose` と `tensorcontract_homspace`

該当:

- `FusionTreeHomSpace::compose`
- `FusionTreeHomSpace::tensorcontract_homspace`

意味:

- `compose`: morphism composition に近い。`lhs.domain` と `rhs.codomain` を直接合成する。
- `tensorcontract_homspace`: 任意軸 contraction を `lhs open / lhs contracted`, `rhs contracted / rhs open` に並べ替え、compose し、最後に output order を適用する。

問題:

- 名前だけだと両者の関係が見えにくい。
- `tensorcontract_homspace` は TensorOperations の `tensorcontract!` 相当の structural lowering であり、単なる homspace composition ではない。

方針:

- `compose` は維持してよい。
- `tensorcontract_homspace` は `contract_homspace_with_index_order` など、axis/index order を伴うことが分かる名前を検討する。

## 残してよい名前

### `tensorcontract_into`, `tensoradd_into`, `tensortrace_into`

Rust では `!` が macro を意味するため、関数名として `tensorcontract!` は使えない。したがって mutating function として `*_into` は妥当。

ただし TensorKit 対応を明記する。

```text
TensorKit tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β)
TeNeT     tensorcontract_into(&mut C, A, B, spec, α, β)
```

将来的には macro facade と expert function を分ける。

```rust
tensorcontract!(c[i; k] += a[i; j] * b[j; k]);
einsum!("i; j, j; k -> i; k", &a, &b)?;
```

### `TensorMap`, `TensorMapSpace`, `FusionTreeHomSpace`, `FusionProductSpace`, `SectorLeg`

TensorKit / categorical tensor の語彙と概ね合っている。維持してよい。

### `DenseExecutor`, `DenseKernelBackend`, `DenseView`

低レイヤ境界名としてはよい。通常の TensorMap tutorial の前面には出さない。

## 実装方針

1. `canonical` を public-facing docs/examples から消す。
2. `TensorContractAxisSpec` を TensorKit の `pA/pB/pAB` に対応する名前へ寄せる。
3. `tree_pair_transform_*` を public 操作名から外し、`permute` / `braid` / `transpose` facade を前面に出す。
4. `explicit_plan`, `canonical_dst`, `dynamic_plan` は execution planning 用語として internal / advanced docs に隔離する。
5. ユーザー層は `einsum!` / `tensorcontract!` macro または builder へ寄せる。
6. 低レイヤ名を変える前に、TensorKit の対応関数列を明記した compatibility table を作る。

## 注意

命名の目的は「Rust らしく見せる」ことではない。TensorKit と同じ意味論を持つ操作が、TeNeT でも同じ概念として理解できることが目的である。

Rust の所有権、cache、workspace、backend 境界は必要だが、それらは public mathematical API 名に混ぜない。
