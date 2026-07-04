# ユーザー層 API 設計メモ

作成日: 2026-07-04。前提: `tensorkit_compatibility_table.md` の P0-P2 実施済み。

## 決定事項(2026-07-04 合意)

1. **TensorKit 準拠**: 関数名・概念は TensorKit の語彙に 1:1 で寄せる。
2. **Runtime は「一度作って以後は暗黙」**: `Runtime` を明示的に構築し、
   `Tensor` が `Arc<Runtime>` を保持。演算はオペランドの runtime を使う。
   日常コードに context 引数は現れない。低レイヤの明示 context API は
   expert 層としてそのまま残す。
3. **メソッド API 先行**: `tensor!` / `einsum!` マクロは後付け。下のメソッド
   API が固まってから薄い糖衣として実装する。

## 目標ユーザーコード

```rust
use tenet::prelude::*;

// Runtime: backend / device / cache 方針をここで一度だけ決める
let rt = Runtime::builder().build()?;            // 既定 CPU バックエンド
// let rt = Runtime::builder().cuda(0).build()?; // GPU (T19 系が繋がり次第)

// 空間: TensorKit の V = U1Space(-1 => 2, 0 => 3, 1 => 2) 相当
let v = Space::u1([(-1, 2), (0, 3), (1, 2)]);

// V ⊗ V ← V ⊗ V のランダムテンソル
let a = Tensor::rand(&rt, [&v, &v], [&v, &v])?;
let b = Tensor::rand(&rt, [&v, &v], [&v, &v])?;

// 縮約
let c = a.compose(&b)?;                          // 圏論的合成 (A * B)
let d = a.contract(&b, [2, 3], [1, 0])?;         // 任意軸(pAB は既定順)
let e = a.contract_ordered(&b, [2, 3], [1, 0], [1, 0, 2, 3])?; // pAB 指定

// インデックス操作(TensorKit permute/braid/transpose)
let p = c.permute([0, 2], [1, 3])?;
let t = c.transpose()?;
let h = c.adjoint()?;

// 分解(matrixalgebra 層と同名を透過: TensorKit 0.17 = MatrixAlgebraKit 系)
let svd = c.svd_trunc(&Truncation::rank(64))?;   // svd.u, svd.s, svd.vh, svd.error
let (u, s, vh) = c.svd_compact()?;
let (q, r) = c.qr_compact()?;
let (v, _) = c.left_orth()?;
let n0 = c.left_null()?;
let x = c.exp()?;                                // eigh 経由の行列関数

// スカラー演算・ノルム(VectorInterface 相当)
let n = c.norm()?;
let s = c.scale(0.5)?;
let w = c.add(&d, 1.0, -1.0)?;                   // w = c - d
let z = c.inner(&d)?;
```

## 命名規則(2026-07-04 追記)

分解・行列関数の名前は **TensorKit 0.17 の export 一覧 = MatrixAlgebraKit 系に
一致させる**(`svd_trunc`/`svd_compact`/`svd_full`/`svd_vals`、`qr_compact`/
`qr_full`/`qr_null`、`lq_*`、`left_orth`/`right_orth`、`left_null`/`right_null`、
`left_polar`/`right_polar`、`eigh_full`/`eigh_trunc`/`eigh_vals`、`eig_*`、
`exp`、`pinv`)。`tsvd`/`leftorth` 等の旧名は 0.17 の export に存在しないため
**採用しない**(alias も作らない)。tenet-matrixalgebra は既にこの命名なので、
ユーザー層は同名メソッドの透過のみ。インデックス操作も同様に export 一覧
基準: `permute`/`braid`/`transpose`/`twist`/`repartition`、構築系は
`id`/`isomorphism`/`unitary`/`isometry`。

## 型設計

- `Space`: sector→degeneracy の列 + 双対フラグ。`SectorLeg` の薄い高レベル形。
  rule ごとのコンストラクタ(`Space::u1`, `Space::z2`, `Space::su2`,
  `Space::fz2`, 積は `Space::product`)。
- `Tensor`: `{ inner: FusionTensorMap 系(rule 型消去), rt: Arc<Runtime> }`。
  **rank は動的**(const generics はユーザー層に出さない)。scalar は当面
  f64、c64 は FactorScalar generic を内包 enum で吸収。
- `Runtime`: `{ TensorContractFusionExecutionContext + DenseExecutor +
  TreeTransformExecutionContext }` を rule ごとに保持(内部は `Mutex` または
  single-thread 前提の `RefCell`;T12 の並列は backend 内なので粗い lock で
  性能問題なし — 演算 1 回につき lock 1 回)。
- rule の型消去: `Tensor` は `Rule` enum(U1 / Z2 / FZ2 / SU2 / 積)を保持し、
  内部でマッチして具象 rule の expert 層に降ろす。ユーザーは rule 型
  パラメータを見ない。

## 層の関係

```text
ユーザー層   Tensor / Space / Runtime          (このメモ)
expert 層    tensorcontract_into / permute_into / svd_compact ...(既存)
内部実行層   Resolution cache / plan / replay / backend       (既存)
```

ユーザー層は expert 層の呼び出しだけで実装し、内部実行層に直接触れない。

## 実施順

1. `Space` + `Tensor` 構築(rand/zeros/from_blocks)+ `Runtime`
2. 縮約・インデックス操作メソッド(compose/contract/permute/adjoint)
3. 分解・行列関数 wrapper(tsvd/leftorth/rightorth/exp/inv/pinv/norm)
4. tutorial.md をユーザー層ベースに書き直し
5. (後日)`tensor!` マクロ、c64、GPU runtime
