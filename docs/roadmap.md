# TeNeT ロードマップ(2026-07-04 棚卸し)

完成像: fZ2⊠U(1)⊠SU(2) の TN 計算、縮約パス最適化、CPU/GPU 両対応、
TensorKit 並みの使い勝手。最重要軸 = Rust 準拠の保守性・拡張性 × 動的 rank でも速い。

## A. ユーザー層
- [x] fZ2⊠U(1)⊠SU(2) 3 因子積(Space + rule enum + TensorKit 数値クロスチェック)
- [x] c64 スカラー(下層 FactorScalar は対応済み)→ eig_* 公開もここに依存
- [x] tensor! の plan cache(トポロジーキー + ドリフト方針)
- [ ] cotengrust 移植(random-greedy / annealing / dynamic slicing / reconfigure)+ sliced 実行
- [x] 部分トレース(`tensor!` の単一オペランド重複ラベル → `trace_pairs` /
      expert `tensortrace_fusion` に lowering、fZ2 supertrace 含め検証済み)
- [x] twist / id / isomorphism / unitary / isometry(TK 0.17 oracle 検証済み、compose は mul! parity で supertrace twist なし)
- [x] repartition / 脚の折り曲げ(codomain↔domain をまたぐ `permute(codomain_axes, domain_axes)`)
      — 能力は実装済み、TK bend factor を oracle 検証(`tenet-core/src/tests.rs:1704/1847/1873`、
      `tenet/tests/user_api.rs:1605`)。名前付き `repartition(N1,N2)` wrapper のみ未実装(sugar)
- [ ] TK export 未実装(code 直読み確認 2026-07-09、workspace で 0 occurrence):
      **catdomain / catcodomain**(bond 脚の直和連結、MPO/環境成長の実ブロッカー)/
      insertunit / removeunit / insertleg(自明脚の挿入除去)/ ishermitian / project_*
- [x] left_orth / right_orth の positive gauge
- [x] tutorial 書き直し

## B. 実行層(CPU)
- [x] batched GEMM 一本化(tenferro 単一所有 + 縮約/transform とも 1 呼び出し/replay)— 並列方針は tenferro 側の改善待ち
- [ ] transform 側並列(T12 phase 2、TK の _add_*_kernel_threaded 相当)— SU(2) cold transform が実アプリのボトルネック
- [ ] typed/dyn execute_resolution の統合(小、~60 行重複)

## C. GPU
- [x] Runtime::cuda(dev) phase 1(direct 縮約、A100 検証済み)
- [ ] device 側 α/β + inactive scale blocks(#1296 の view accum で表現可能)
- [ ] tree-transform recoupling / scale / strided copy の device kernel
- [ ] dynamic route の device scratch(ScratchStorage::reset_filled の stream-ordered 実装)
- [ ] device 分解(cuSOLVER 経由、DenseExecutor device 配線)
- [ ] CUDA grouped GEMM(stream 並列、#1293 の CUDA 後続 issue)
- [ ] GPU ベンチ(χ スケーリング)、将来 4×A100 マルチ GPU

## D. 外部依存
- [x] tenferro#1296 マージ済み
- [x] tenferro#1297 マージ済み(faer 並列方針の改善提案を #1293 に報告済み)
- [ ] T21: tenferro in-place 分解(svd/qr/eigh into preallocated)issue intake から
- [ ] strided-rs#135 レビュー待ち
- [ ] stage-2 残制限: view 出力 × k=0 × β≠1 の strided scale kernel

## E. 物理応用(= API 検証)
- [x] iTEBD デモ(U(1) Heisenberg 鎖): E/bond = -0.443008 vs 厳密 -0.443147
      (誤差 1.4e-4、chi=32、12.2s/2500 steps)— examples/itebd_heisenberg.rs
- [ ] CTMRG / iPEPS デモ(fZ2xU(1)xSU(2) を使う本命)
- [ ] finite-torus コードの新ユーザー層移行

## F. iTEBD デモで見つかった API 課題(2026-07-04)
- [x] 【重要】leg 次元・適合検査が「populated blocks」由来で、疎な対称状態
      (Neel 積状態等)の正当な縮約を拒否する。根本原因: Tensor が Space
      レベルの sector 内容を保持していない。→ SectorLeg が per-sector
      degeneracy を保持(TensorKit GradedSpace パリティ)、leg_dims /
      結果 space 補完 / network 検査を leg 基準化、エラーは両 leg の
      (sector, deg, dual) とオペランド番号・ラベルを表示
- [x] tensor! のプラン再利用(topology key + drift-factor replanning。
      truncation sweep では同じ topology を再利用、必要時のみ再計画)
- [ ] tensor! で field access オペランド(svd.u[..])がパース不可(要括弧)
- [ ] 参考: λ⁻¹ は pinv で clean に書けた、exp() は rank-4 endomorphism に
      直接効いた(追加実装不要だった良い所見)

## 完了済み(2026-07-03〜04)
coupled-sector レイアウト一本化 / canonical 直 GEMM(mul! パリティ)/
GEMM recoupling(T20)/ resolution cache 統合(T18 + MRU ring)/
batched-GEMM seam(T12 phase 0)/ GPU 縦切り A100 実証(T19)/
命名監査 P0-P2 / 動的 rank ユーザー層 / 分解メソッド一式 / tensor! マクロ。
TensorKit との直接対決は全 24 項目で同等以上。
