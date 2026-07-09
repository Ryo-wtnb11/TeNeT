# ユーザー層 API 設計メモ

作成日: 2026-07-04。前提: `tensorkit_compatibility_table.md` の P0-P2 実施済み。

## 決定事項(2026-07-04 合意)

1. **TensorKit 準拠**: 関数名・概念は TensorKit の語彙に 1:1 で寄せる。
2. **Runtime は「一度作って以後は暗黙」**: `Runtime` を明示的に構築し、
   `Tensor` が `Arc<Runtime>` を保持。演算はオペランドの runtime を使う。
   日常コードに context 引数は現れない。低レイヤの明示 context API は
   expert 層としてそのまま残す。
3. **メソッド API 先行**: `tensor!` マクロは後付け。下のメソッド API が
   固まってから薄い糖衣として実装する。
4. **記法は @tensor 準拠、einsum 文字列は公開しない**(2026-07-04 合意):
   ユーザー面は識別子インデックスの proc-macro
   `tensor!(c[a, b; g, h] = x[a, b; i, j] * y[i, j; g, h])`。
   ラベルは legacy planner の NetworkIR に直接下ろす(文字列 einsum
   パーサは公開 API に出さない)。N 体は planner(greedy/cotengrust)が
   順序を自動選択。

## 目標ユーザーコード

```rust
use tenet::prelude::*;
use tenet_network::tensor;

// Runtime: backend / device / cache 方針をここで一度だけ決める
let rt = Runtime::builder().build()?;            // 既定 CPU バックエンド
// let rt = Runtime::builder().cuda(0).build()?; // CUDA feature 有効時

// 空間: TensorKit の V = U1Space(-1 => 2, 0 => 3, 1 => 2) 相当
let v = Space::u1([(-1, 2), (0, 3), (1, 2)]);

// V ⊗ V ← V ⊗ V のランダムテンソル
let a = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v, &v])?;
let b = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v, &v])?;
let z = Tensor::rand(&rt, Dtype::C64, [&v, &v], [&v, &v])?; // c64 は dtype token で明示構築

// 縮約
let c = a.compose(&b)?;                          // 圏論的合成 (A * B)
let d = a.contract(&b, &[2, 3], &[1, 0])?;       // 任意軸(pAB は既定順)
let e = a.contract_ordered(&b, &[2, 3], &[1, 0], &[1, 0, 2, 3])?; // pAB 指定
let f = tensor!([i, j; g, h] = a[i, j; k, l] * b[k, l; g, h])?;

// インデックス操作(TensorKit permute/braid/transpose)
let p = c.permute(&[0, 2], &[1, 3])?;
let t = c.transpose()?;
let h = c.adjoint()?;

// 分解(matrixalgebra 層と同名を透過: TensorKit 0.17 = MatrixAlgebraKit 系)
let svd = c.svd_trunc(&Truncation::rank(64))?;   // svd.u, svd.s, svd.vh, svd.error
let (u, s, vh) = c.svd_compact()?;
let (q, r) = c.qr_compact()?;
let (iso, _) = c.left_orth()?;
let n0 = c.left_null()?;
let (d_eig, w_eig) = c.eig_full()?;              // d_eig/w_eig は常に c64
let x = c.exp()?;                                // eigh 経由の行列関数

// スカラー演算・ノルム(VectorInterface 相当)
let n = c.norm()?;
let s = c.scale(0.5)?;
let w = c.add(&d, 1.0, -1.0)?;                   // w = c - d
let ip: Scalar = c.inner(&d)?;                   // variant は入力 dtype に一致 (F64/C64)
let c_c64 = c.to_c64();                          // c64 には明示 widening
```

## 命名規則(2026-07-04 追記)

分解・行列関数の名前は **TensorKit 0.17 の export 一覧 = MatrixAlgebraKit 系に
一致させる**(`svd_trunc`/`svd_compact`/`svd_full`/`svd_vals`、`qr_compact`/
`qr_full`/`qr_null`、`lq_*`、`left_orth`/`right_orth`、`left_null`/`right_null`、
`left_polar`/`right_polar`、`eigh_full`/`eigh_trunc`/`eigh_vals`、`eig_full`/
`eig_trunc`/`eig_vals`、`exp`、`pinv`)。tenet-matrixalgebra は既にこの命名なので、
ユーザー層は同名メソッドの透過のみ。0.17 の export にない legacy alias は
作らない。インデックス操作も同様に export 一覧基準:
`permute`/`braid`/`transpose`/`twist`、構築系は
`id`/`isomorphism`/`unitary`/`isometry`。repartition(脚の折り曲げ)は独立
メソッドではなく `permute(codomain_axes, domain_axes)` に codomain↔domain を
またぐ分割を渡す形で提供済み(名前付き wrapper は未実装)。未実装の TK export:
catdomain / catcodomain / insertunit / removeunit / insertleg / ishermitian /
project_*(roadmap.md 参照)。

## 縮約適合性の契約(2026-07-04 合意)

- **index object は作らない**(ITensor 方式は採らない)。脚は位置で指定し、
  ラベルは `tensor!` 式内の一時的な束縛。
- **適合性は oriented object の双対対で担保する**(TensorKit 方式):
  codomain 脚はその Space、domain 脚はその双対として解釈し、縮約で
  つながる 2 脚の oriented object が互いに双対であること。通常の
  codomain-vs-domain 合成では同じ `Space` 同士が縮約できる。同じ側の
  2 脚を縮約する場合は、実際の `Space` の片方だけを `dual()` にする。
  違反は実行時の型付きエラー(leg duality / space mismatch)。
- `tensor!` マクロはラベルの整合(片側にしか現れない添字、重複等)を
  コンパイル時に検査できる。Space の一致は実行時検査(低レイヤの既存
  ゲートがそのまま担う)。

## 型設計

- `Space`: sector→degeneracy の列 + 双対フラグ。`SectorLeg` の薄い高レベル形。
  rule ごとのコンストラクタ(`Space::u1`, `Space::z2`, `Space::su2`,
  `Space::fz2`, 積は `Space::product`)。
- `Tensor`: `{ inner: FusionTensorMap 系(rule 型消去), rt: Arc<Runtime> }`。
  **rank は動的**(const generics はユーザー層に出さない)。storage dtype は
  `f64`/`c64` の enum。混在演算は昇格せず `DtypeMismatch`、必要なら
  `to_c64()` で明示 widening。
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
3. 分解・行列関数 wrapper(`svd_trunc`/`left_orth`/`right_orth`/`exp`/`inv`/
   `pinv`/`norm`/`eig_*`)
4. tutorial.md をユーザー層ベースに書き直し
5. legacy planner (tenet-legacy/tenet-contract の構造半分 + tenet-cotengrust)
   を移植し、`tensor!` proc-macro(@tensor 記法)を NetworkIR 直結で実装。
   execution 半分のみ新ユーザー層 Tensor に再結線
   → 実施済み(2026-07-04): `tenet-network`(planner 移植 + Tensor executor)
   + `tenet-macros`(`tensor!`)。構文は式形式
   `let c = tensor!([a, b; g, h] = x[a, b; i, j] * y[i, j; g, h])?;`
   (出力シグネチャ先頭、`conj(x)[...]` で adjoint、`[]` で rank-0)。
   フォローアップ: (i) tenet-cotengrust 移植(DenseContractionOptimizer
   実装をそのまま差し込む)、(ii) Tensor 層 select_index が入り次第
   sliced executor。
6. 実施済み(2026-07-04): c64、`eig_*`、topology-keyed plan cache、
   partial trace、CUDA phase 1(direct f64 contraction)。
