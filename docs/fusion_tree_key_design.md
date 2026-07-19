> **Current implementation boundary (issue #322, N3a):** tree-transform plans,
> structures, and recoupling rows are reused only in validated process-resident
> caches. Cache bounds remain separate policy work; retiring disk persistence
> does not claim or introduce a new bound. The former automatic v1/v2 disk
> execution-plan cache has been retired, so stale `tree_transform_plans_v1.bin` and
> `tree_transform_plans_v2.bin` files are ignored and may be deleted manually.
> The persistent-cache sections below are historical design exploration, not
> the current implementation plan. Explicit network contraction-order
> persistence through `tenet-network::{save_plan_cache, load_plan_cache}` is a
> separate application-owned feature and remains supported.

以下では **best な最終設計** と **そこへ向かう実装計画** を明確に分けます。ここでは将来の対象を **現在の multiplicity-free だけでなく、Generic fusion、つまり multiplicity あり** まで含むものとして扱います。

---

# 1. まず結論

TeNeT の最終設計として最も良いのは、**`FusionTreeKey` を速くする設計ではありません**。  
最も良いのは、**hot path から `FusionTreeKey` を消す設計**です。

つまり、最終形はこうです。

```rust
FusionTreeSpec      // semantic representation: 意味・debug・serialization 用
TreeArena           // canonical storage: 融合木を一意化して保存
TreeId              // runtime representation: 計算中に持ち回る軽い id
TreePairId          // recoupling cache / accumulator 用の key
CompactKey          // arena lookup 用の exact compact key
```

現在の `FusionTreeKey` は semantic key と runtime key を兼ねています。  
これが問題です。

現在の構造は、

```rust
pub struct FusionTreeKey {
    uncoupled:  SectorVec,
    coupled:    Option<SectorId>,
    is_dual:    DualVec,
    innerlines: SectorVec,
    vertices:   SectorVec,
}
```

で、`vertices` は `Hash/Eq/Ord` から除外されています。これは現行の multiplicity-free 前提では正しいです。TeNeT のコードコメントにも、multipity-free では vertices が常に trivial label なので identity から除外している、と書かれています。([raw.githubusercontent.com](https://raw.githubusercontent.com/Ryo-wtnb11/TeNeT/ac636fefb5864a1f6e5bf843cd419f958494da7e/tenet-core/src/fusion_tree.rs))

しかし、将来 **multiplicity あり**まで行くなら、`vertices` は identity に入る必要があります。TensorKit 側も `FusionTree` に `vertices::NTuple{L,Int}` を持ち、multiplicity-free では constant `1`、Generic fusion では vertex label として扱っています。さらに TensorKit の hash/eq は Generic fusion では `vertices` を含めます。([raw.githubusercontent.com](https://raw.githubusercontent.com/QuantumKitHub/TensorKit.jl/v0.17.0/src/fusiontrees/fusiontrees.jl))

したがって最終的な方針はこれです。

> **FusionTreeKey を小さくするのではなく、fusion tree を canonical arena に置き、hot path では `TreeId` / `TreePairId` だけを使う。**  
> **vertices は消さない。multiplicity-free では implicit trivial、Generic fusion では explicit label として path identity に含める。**

---

# 2. 現状の正しい診断

現状の TeNeT は、かなり良いところまで来ています。

TeNeT の `SectorId` は `usize` newtype で、`FusionStyleKind` には `Unique`, `Simple`, `Generic` があり、`Generic` だけが multiplicity を持つ設計になっています。([raw.githubusercontent.com](https://raw.githubusercontent.com/Ryo-wtnb11/TeNeT/ac636fefb5864a1f6e5bf843cd419f958494da7e/tenet-core/src/sector.rs))  
また `SectorVec` と `DualVec` は `SmallVec<[...; 8]>` で、inline capacity 8 は小ランクの metadata を allocation-free にするためのものです。([raw.githubusercontent.com](https://raw.githubusercontent.com/Ryo-wtnb11/TeNeT/ac636fefb5864a1f6e5bf843cd419f958494da7e/tenet-core/src/storage.rs))

つまり今の設計の問題は、

```text
SmallVec が悪い
```

ではありません。

本当の問題は、

```text
semantic key である FusionTreeKey を、
recoupling cold path の runtime key として何度も clone/hash/eq している
```

ことです。

実際、`TreePairRowMemo` は現在、

```rust
FxHashMap<(RuleKey, TreeTransformOperation, FusionTreeBlockKey), Arc<...>>
```

のような形で、`FusionTreeBlockKey` を memo key にしています。memo pre-pass でも `src_key.clone()` が出てきます。([raw.githubusercontent.com](https://raw.githubusercontent.com/Ryo-wtnb11/TeNeT/ac636fefb5864a1f6e5bf843cd419f958494da7e/tenet-tensors/src/tree_transform/plan.rs))  
また process-global cache / row memo は `RwLock<FxHashMap<...>>` 系で持たれており、persistent cache もすでに存在します。([raw.githubusercontent.com](https://raw.githubusercontent.com/Ryo-wtnb11/TeNeT/ac636fefb5864a1f6e5bf843cd419f958494da7e/tenet-tensors/src/tree_transform/cache.rs))

したがって、ここにさらに naïve な global interner を足すと、**キーは軽くなるが lock 系がさらに増える**危険があります。

---

# 3. Best な最終設計

## 3.1 Semantic model を分解する

最終的な fusion tree は、次の 3 層に分けるのがよいです。

```text
ExternalSpace = uncoupled + is_dual
TreeSpace     = ExternalSpace + coupled
FusionPath    = innerlines + vertices
TreeId        = TreeSpaceId + PathId
```

ここで重要なのは、`coupled` を `ExternalSpace` に入れないことです。

TensorKit でも、同じ block の中では `uncoupled` と `isdual` が共通なので、tree index ではそれらを避け、`coupled`, `innerlines`, `vertices` 側を使う構造になっています。([raw.githubusercontent.com](https://raw.githubusercontent.com/QuantumKitHub/TensorKit.jl/v0.17.0/src/fusiontrees/fusiontrees.jl))

TeNeT でも同じ思想にすべきです。

```rust
#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ExternalSpaceId(u32);

#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TreeSpaceId(u32);

#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PathId(u32);

#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TreeId {
    pub space: TreeSpaceId,
    pub path: PathId,
}
```

あるいは hot path では packed にしてよいです。

```rust
#[repr(transparent)]
#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TreeId(u64);

// high 32 bits = TreeSpaceId
// low  32 bits = PathId
```

`TreePairId` はこうです。

```rust
#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TreePairId {
    pub codomain: TreeId,
    pub domain: TreeId,
}
```

これで `clone/eq/hash` は **integer operation** になります。

---

## 3.2 `coupled: Option<SectorId>` は runtime key から外す

現在の `FusionTreeKey` は `coupled: Option<SectorId>` です。  
しかし canonical runtime tree では、`coupled` は基本的に **必ず存在する値**にした方がよいです。

つまり best design では、

```rust
pub struct TreeSpaceKey {
    pub external: ExternalSpaceId,
    pub coupled: SectorId,
}
```

とする。

`Option<SectorId>` は legacy API、partial tree、または construction helper のために残すのはよいです。  
しかし **arena に intern される canonical tree** では `coupled` を non-optional にする。

理由は単純です。

```text
None を持つ tree と Some(vacuum) を持つ tree が混ざると、
canonical identity が曖昧になる
```

からです。

---

## 3.3 multiplicity ありを見据えた `FusionPath`

ここが一番重要です。

現在の multiplicity-free では、

```text
PathKey = innerlines
vertices = always trivial
```

で済みます。

しかし Generic fusion では、

```text
PathKey = innerlines + vertices
```

です。

`vertices` は sector ではありません。  
したがって `SectorId` ではなく、別 newtype にするべきです。

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct VertexId(u32);
```

`u16` でも足りる可能性は高いですが、ライブラリの public core 型としては `u32` の方が安全です。packing 時だけ小さくすればよいです。

path label は fusion style に応じて変わります。

```rust
pub enum PathLabelSpec {
    Unique,
    Simple {
        innerlines: Box<[SectorId]>,
    },
    Generic {
        innerlines: Box<[SectorId]>,
        vertices: Box<[VertexId]>,
    },
}
```

ただし arena 内では enum を per path に持つより、`PathArena` が fusion style を持つ方がよいです。

```rust
pub struct PathArena {
    style: FusionStyleKind,

    inner_len: u16,
    vertex_len: u16,

    // path_id * inner_len .. path_id * inner_len + inner_len
    innerlines: Vec<SectorId>,

    // only for Generic
    vertices: Vec<VertexId>,

    path_map: FxHashMap<CompactPathKey, PathId>,
}
```

style ごとの意味はこうです。

| fusion style | path identity | physical storage |
|---|---|---|
| **Unique** | no independent path | usually one path, no innerlines/vertices |
| **Simple / multiplicity-free but multiple outputs** | `innerlines` | store innerlines only |
| **Generic / multiplicityあり** | `innerlines + vertices` | store both |

これは TensorKit の方針とも合います。TensorKit は `MultipleFusion` なら `innerlines` を hash に含め、`GenericFusion` ならさらに `vertices` を hash に含めています。([raw.githubusercontent.com](https://raw.githubusercontent.com/QuantumKitHub/TensorKit.jl/v0.17.0/src/fusiontrees/fusiontrees.jl))

---

## 3.4 `vertices` は「消す」のではなく「implicit trivial」にする

将来 Generic fusion まで行くなら、

```text
vertices を FusionTreeKey から完全に消す
```

のは悪手です。

正しい設計はこれです。

```rust
pub enum VertexLabels {
    Trivial,
    Explicit(Box<[VertexId]>),
}
```

ただし最終 arena では、per path に `VertexLabels` を持つより、`PathArena` 全体で style を持つ方が軽いです。

```rust
pub struct PathArena {
    style: FusionStyleKind,
    inner_len: u16,
    vertex_len: u16,
    innerlines: Vec<SectorId>,
    vertices: Vec<VertexId>, // empty unless Generic
}
```

multiplicity-free では `vertices` storage はゼロ。  
Generic fusion では `vertices` を identity に含める。

これが best です。

---

## 3.5 TreeArena は category-scoped にする

global static interner は避けるべきです。

悪い例はこれです。

```rust
static GLOBAL_TREE_INTERNER: RwLock<HashMap<FusionTreeSpec, TreeId>>;
```

理由は、fusion tree identity は sector id だけでは決まらないからです。

同じ `SectorId` 列でも、次が違えば recoupling coefficient は変わります。

```text
fusion rule
dual convention
F-symbol gauge
R-symbol convention
vertex basis convention
```

したがって、arena は **category-scoped** にするべきです。

```rust
pub struct FusionCategoryCtx {
    pub category_id: CategoryId,
    pub fingerprint: CategoryFingerprint,
    pub tree_arena: TreeArena,
}
```

`CategoryFingerprint` は少なくとも次を含むべきです。

```rust
pub struct CategoryFingerprint {
    pub sector_registry_hash: u64,
    pub fusion_rule_hash: u64,
    pub dual_rule_hash: u64,
    pub f_symbol_hash: u64,
    pub r_symbol_hash: u64,
    pub vertex_basis_hash: u64,
}
```

特に Generic fusion では **vertex basis gauge** が重要です。  
同じ labels でも vertex basis の取り方が違えば F-symbol matrix が変わります。

---

## 3.6 TreeArena の構造

最終形はこうです。

```rust
pub struct TreeArena {
    external_spaces: Vec<ExternalSpaceRecord>,
    external_map: FxHashMap<CompactExternalSpaceKey, ExternalSpaceId>,

    tree_spaces: Vec<TreeSpaceRecord>,
    tree_space_map: FxHashMap<CompactTreeSpaceKey, TreeSpaceId>,
}

pub struct ExternalSpaceRecord {
    pub rank: u16,
    pub uncoupled: Box<[SectorId]>,
    pub is_dual_bits: Box<[u64]>, // bool slice より compact
}

pub struct TreeSpaceRecord {
    pub external: ExternalSpaceId,
    pub coupled: SectorId,
    pub paths: PathArena,
}
```

`PathArena` は flat SoA が良いです。  
SoA は **structure of arrays** の略で、構造体の配列ではなく、各 field を別々の連続配列に置く方式です。

```rust
pub struct PathArena {
    pub style: FusionStyleKind,

    pub inner_len: u16,
    pub vertex_len: u16,

    pub innerlines: Vec<SectorId>,
    pub vertices: Vec<VertexId>,

    pub path_map: FxHashMap<CompactPathKey, PathId>,
}
```

`innerlines` の取り出しは、

```rust
let start = path_id as usize * inner_len;
let end = start + inner_len;
&innerlines[start..end]
```

です。

rank は dynamic のままです。  
しかし path storage は fixed-stride になります。  
これは `SmallVec` を大量に持つより allocation が少なく、cache-friendly です。

---

## 3.7 Runtime cache key

recoupling row memo の最終 key は、概念的にはこうです。

```rust
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct RowMemoKey {
    pub category: CategoryId,
    pub operation: OperationId,
    pub src: TreePairId,
}
```

`RuleKey` を残すなら、

```rust
pub struct RowMemoKey<RuleKey> {
    pub rule: RuleKey,
    pub operation: OperationId,
    pub src: TreePairId,
}
```

でもよいです。

ただし、operation も clone が重いなら intern した方がよいです。

```rust
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct OperationId(u32);
```

現在は memo key に `TreeTransformOperation` と `FusionTreeBlockKey` が入っています。([raw.githubusercontent.com](https://raw.githubusercontent.com/Ryo-wtnb11/TeNeT/ac636fefb5864a1f6e5bf843cd419f958494da7e/tenet-tensors/src/tree_transform/plan.rs))  
最終形では、ここを

```text
RuleKey + OperationId + TreePairId
```

に落とす。

これで row memo lookup はほぼ integer key になります。

---

# 4. interning の是非

## 4.1 interning は必要。ただし global static ではない

結論は、

```text
interning は yes
global static interning は no
category-scoped arena + plan-local arena が best
```

です。

### なぜ global static が悪いか

global static interner は簡単に見えます。

```rust
static INTERNER: RwLock<FxHashMap<FusionTreeKey, TreeId>>;
```

しかし問題が多いです。

- category が違う tree を混ぜる危険がある
- lock contention が増える
- persistent cache と id の整合が難しい
- eviction するなら generation が必要
- eviction しないなら memory が増え続ける
- parallel cold compile で id assignment が非決定的になりやすい

TeNeT にはすでに global `RwLock` 系の plan/structure/row cache と persistent cache があるので、さらに global interner を足すのは設計として重いです。([raw.githubusercontent.com](https://raw.githubusercontent.com/Ryo-wtnb11/TeNeT/ac636fefb5864a1f6e5bf843cd419f958494da7e/tenet-tensors/src/tree_transform/cache.rs))

---

## 4.2 best は 2 層 interning

最終的にはこの 2 層がよいです。

```text
Category-level TreeArena:
    長寿命。
    TreeId を process 内で安定にする。
    global row memo と共有できる。

Plan-local DenseArena:
    短寿命。
    plan compile / compose_block_terms の hot path 用。
    lock-free。
    dense index / sort-reduce に使う。
```

具体的には、

```rust
pub struct PlanningContext<'a> {
    pub category: &'a FusionCategoryCtx,
    pub local: LocalTreeIndex,
}
```

local 側では、

```rust
#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct LocalTreeId(u32);

#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct LocalTreePairId(u32);
```

を使う。

最終的に plan を publish するときに、

```text
LocalTreeId -> TreeId
LocalTreePairId -> TreePairId
```

へ対応づける。

これにより、hot loop では lock を踏まず、global memo では stable `TreeId` を使えます。

---

## 4.3 generation 管理

append-only arena なら generation は不要です。

```rust
pub struct TreeId(u64);
```

で十分です。

ただし、tree arena 自体を LRU eviction するなら、

```rust
pub struct TreeId {
    index: u32,
    generation: u32,
}
```

が必要です。

私の推奨は、**tree arena は eviction しない**ことです。  
evict するのは row memo / plan cache / dense plan だけでよいです。

理由は、tree spec は row data や dense plan に比べれば小さいからです。  
tree id の安定性を失う方が痛いです。

---

# 5. bit packing の評価

## 5.1 full `FusionTreeKey -> u128` は best ではない

rank 8 の tree でも、identity に必要な情報は多いです。

multiplicity-free でも概念的には、

```text
uncoupled:  8
coupled:    1
innerlines: 6
is_dual:    8 bits
```

です。

Generic fusion ならさらに、

```text
vertices: 7
```

が入ります。

つまり rank 8 でも、

```text
sector ids: 15
vertex ids: 7
dual bits: 8
```

になります。

`u128` に exact encoding するにはかなり強い仮定が必要です。

```text
SectorId は常に 8 bit 以下
VertexId も小さい
rank tag も format tag も不要
```

のような前提が必要になります。  
これは library core の設計として硬すぎます。

---

## 5.2 bit packing は arena lookup 用に使う

bit packing の正しい位置はここです。

```text
FusionTreeKey の主表現
```

ではなく、

```text
TreeArena の map key
```

です。

例:

```rust
pub enum CompactPathKey {
    Small(SmallPathKey),
    Large(Arc<[u8]>),
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct SmallPathKey {
    pub len_inner: u8,
    pub len_vertex: u8,
    pub words: [u64; 4],
}
```

あるいは、

```rust
#[derive(Clone, Eq, PartialEq, Hash)]
pub enum CompactPathKey {
    Inline {
        len_inner: u8,
        len_vertex: u8,
        bytes: [u8; 64],
    },
    Heap(Box<[u8]>),
}
```

この encoding は **exact** であるべきです。  
fingerprint-only key にしてはいけません。

Hash collision は HashMap 側で起きてもよいですが、`Eq` は exact でなければだめです。

---

## 5.3 Ord は packed bytes に任せない

`Ord` を packed bytes に任せるのは危険です。

たとえば length を先頭に入れると、

```text
[2]
[1, 999]
```

の順序が semantic slice order と変わる可能性があります。

したがって、

```text
CompactKey は HashMap lookup 用
public / deterministic order は semantic order or canonical id order
```

に分けるべきです。

数値結果を bit 一致レベルで守るなら、row output order は特に重要です。  
parallel compile で id assignment が race によって変わると、addition order が変わり、floating-point result が変わる可能性があります。

したがって最終設計では、

```text
PathId assignment は deterministic
row entries は deterministic order に sort
same key の coefficient reduction も deterministic
```

にするべきです。

---

# 6. fixed inline buffer の評価

`[SectorId; MAX] + len` は、主設計としては弱いです。

たとえば、

```rust
struct FixedSectorSeq<const MAX: usize> {
    len: u8,
    data: [SectorId; MAX],
}
```

にすれば rank 9〜15 の heap spill は消えます。

しかし rank 2〜6 の key まで太ります。  
現在は `SmallVec<[SectorId; 8]>` により、小 rank では heap を踏まない設計です。([raw.githubusercontent.com](https://raw.githubusercontent.com/Ryo-wtnb11/TeNeT/ac636fefb5864a1f6e5bf843cd419f958494da7e/tenet-core/src/storage.rs))

つまり fixed inline は、

```text
rank > 8 の spill を消す
代わりに common small key の size を増やす
```

という trade-off です。

これは **局所最適**です。  
best design ではありません。

実験として、

```rust
SectorVec = SmallVec<[SectorId; 12]>
SectorVec = SmallVec<[SectorId; 16]>
```

を feature flag で測るのはありです。  
ただし最終設計の主軸にはしない方がよいです。

---

# 7. cached hash の評価

`FusionTreeKey` に cached fingerprint を持たせるのは、低リスクの補助策です。

```rust
pub struct FusionTreeSpec {
    pub external: ExternalSpaceSpec,
    pub coupled: SectorId,
    pub path: FusionPathSpec,
    pub fingerprint: u64,
}
```

ただし fingerprint は identity ではありません。

```rust
impl PartialEq for FusionTreeSpec {
    fn eq(&self, other: &Self) -> bool {
        self.fingerprint == other.fingerprint
            && self.external == other.external
            && self.coupled == other.coupled
            && self.path == other.path
    }
}
```

`Hash` は fingerprint を使ってよいですが、`Eq` は full compare を残すべきです。

これは hash self-time には効きます。  
しかし clone cost は本質的には残ります。

したがって best design では、

```text
cached hash は補助
TreeId 化が本命
```

です。

---

# 8. block-local representation

ここはかなり重要です。

同じ tree block 内では、多くの場合、

```text
uncoupled
is_dual
```

が共通です。

TensorKit も同じ理由で、block 内の tree index では `uncoupled` と `isdual` を避けています。([raw.githubusercontent.com](https://raw.githubusercontent.com/QuantumKitHub/TensorKit.jl/v0.17.0/src/fusiontrees/fusiontrees.jl))

TeNeT の best design でも、

```text
ExternalSpaceId = uncoupled + is_dual
TreeIndexData   = coupled + path
```

にするべきです。

つまり block 内の dense map はこうです。

```rust
#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct BlockTreeIndexKey {
    pub coupled: SectorId,
    pub path: PathId,
}
```

tree pair なら、

```rust
#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct BlockTreePairIndexKey {
    pub codomain_coupled: SectorId,
    pub codomain_path: PathId,
    pub domain_path: PathId,
}
```

もし codomain/domain の coupled が常に同じなら、`coupled` は 1 回だけでよいです。

これにより、block 内では full `FusionTreeKey` を hash しません。

---

# 9. compose / accumulator の best design

現在の accumulator は key を `HashMap` に入れ、vacant entry で key clone が出ます。([raw.githubusercontent.com](https://raw.githubusercontent.com/Ryo-wtnb11/TeNeT/ac636fefb5864a1f6e5bf843cd419f958494da7e/tenet-core/src/fusion_tree.rs))

最終形では、accumulator の key は `TreePairId` または local dense id にします。

```rust
pub enum FusionTermAccumulator<S> {
    Empty,
    Singleton(TreePairId, S),
    Map {
        order: Vec<TreePairId>,
        coefficients: FxHashMap<TreePairId, S>,
    },
}
```

さらに良いのは、term 数が多い場合に **sort/reduce** へ切り替えることです。

```rust
Vec<(TreePairId, S)>
```

へ push して、

```text
sort by TreePairId
same key を deterministic order で reduce
```

する。

`TreePairId` が integer なら sort は速いです。  
HashMap より deterministic にしやすいです。

best design では、accumulation strategy はこう分けます。

| case | strategy |
|---|---|
| very few terms | singleton / small vec |
| sparse medium terms | `FxHashMap<TreePairId, S>` |
| many terms / block batched | `Vec<(TreePairId,S)>` + sort/reduce |
| fully enumerated block | dense `Vec<S>` by dense pair index |

---

# 10. persistent cache との整合

persistent cache に `TreeId` をそのまま保存してはいけません。

理由は、`TreeId` は process 内の arena id だからです。  
arena の構築順が変わると、同じ tree でも id が変わります。

disk に保存すべきなのは semantic data です。

```text
cache magic
cache version
category fingerprint
operation spec
external space specs
coupled sectors
path specs
row entries
scalar data
```

load 時に、

```text
semantic spec を読む
category fingerprint を検証
current TreeArena に再 intern
新しい TreeId を得る
row data を結び直す
```

という流れにする。

TeNeT の persistent cache はすでに versioned file として存在しているので、この方針は自然です。([raw.githubusercontent.com](https://raw.githubusercontent.com/Ryo-wtnb11/TeNeT/ac636fefb5864a1f6e5bf843cd419f958494da7e/tenet-tensors/src/tree_transform/cache.rs))

---

# 11. Generic fusion で変わること

multiplicity-free では F/R-symbol は多くの場合 scalar 的に扱えます。  
しかし Generic fusion では、recoupling は vertex basis 間の **matrix** になります。

そのため row は自然に sparse row になります。

```rust
pub struct RecouplingRow<S> {
    pub entries: Vec<(TreePairId, S)>,
}
```

Generic fusion の row key は少なくとも、

```text
category / rule
operation
source tree pair
```

を含む必要があります。

そして destination 側は、

```text
target TreePairId + coefficient
```

の列になります。

重要なのは、Generic fusion では、

```text
same innerlines but different vertices
```

が別の basis state になることです。

したがって `vertices` を hash/eq から除外する設計は、最終的には不可です。  
現在は multiplicity-free だから除外してよいだけです。

---

# 12. Best design の全体像

最終形をまとめると、こうです。

```rust
pub struct FusionCategoryCtx {
    pub category_id: CategoryId,
    pub fingerprint: CategoryFingerprint,
    pub tree_arena: TreeArena,
}

pub struct TreeArena {
    pub external_spaces: Vec<ExternalSpaceRecord>,
    pub external_map: FxHashMap<CompactExternalSpaceKey, ExternalSpaceId>,

    pub tree_spaces: Vec<TreeSpaceRecord>,
    pub tree_space_map: FxHashMap<CompactTreeSpaceKey, TreeSpaceId>,
}

pub struct ExternalSpaceRecord {
    pub rank: u16,
    pub uncoupled: Box<[SectorId]>,
    pub is_dual_bits: Box<[u64]>,
}

pub struct TreeSpaceRecord {
    pub external: ExternalSpaceId,
    pub coupled: SectorId,
    pub paths: PathArena,
}

pub struct PathArena {
    pub style: FusionStyleKind,
    pub inner_len: u16,
    pub vertex_len: u16,

    pub innerlines: Vec<SectorId>,
    pub vertices: Vec<VertexId>,

    pub path_map: FxHashMap<CompactPathKey, PathId>,
}

#[repr(transparent)]
#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TreeId(u64);

#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TreePairId {
    pub codomain: TreeId,
    pub domain: TreeId,
}
```

Semantic representation は別に持ちます。

```rust
pub struct FusionTreeSpec {
    pub external: ExternalSpaceSpec,
    pub coupled: SectorId,
    pub path: FusionPathSpec,
}

pub struct ExternalSpaceSpec {
    pub uncoupled: Box<[SectorId]>,
    pub is_dual: Box<[bool]>,
}

pub enum FusionPathSpec {
    Unique,
    Simple {
        innerlines: Box<[SectorId]>,
    },
    Generic {
        innerlines: Box<[SectorId]>,
        vertices: Box<[VertexId]>,
    },
}
```

`FusionTreeKey` は最終的にはこの semantic type の legacy wrapper にする。

```text
FusionTreeKey:
    external API
    debug
    serialization
    parity tests
    compatibility

TreeId:
    recoupling
    cache lookup
    accumulator
    block transform
```

---

# 13. 各案への最終評価

| 案 | best design での位置づけ | 評価 |
|---|---|---|
| **global interning** | 不採用。category-scoped にする | global lock と category 混線が危険 |
| **category-scoped arena** | 採用 | TreeId を process 内で安定化 |
| **plan-local arena** | 採用 | hot path を lock-free にする |
| **u128 packing** | 不採用 | full tree には狭すぎる |
| **fixed byte packing** | 採用。ただし arena lookup 用 | exact CompactKey として有効 |
| **fixed `[SectorId; MAX]`** | 主軸ではない | rank > 8 対策にはなるが小 key が太る |
| **cached hash** | 補助 | 低リスクだが clone 問題は残る |
| **SoA arena** | 採用 | allocation と cache footprint を下げる |
| **block-local key** | 採用 | `uncoupled/is_dual` を繰り返し hash しない |
| **perfect hashing / dense id** | 採用候補 | TreeId 化後に効く |
| **sort/reduce accumulator** | 採用候補 | deterministic で cache-friendly |

---

# 14. 実装計画

ここからは **best design へ向けた移行計画**です。  
これは実装上の段階論です。best design そのものとは分けます。

---

## Phase 0: 測定を固定する

まず変更前の baseline を固定する。

最低限これを取るべきです。

```text
cold total time
warm total time
FusionTreeKey clone self-time
FusionTreeKey hash self-time
FusionTreeKey eq self-time
allocation count
rank histogram
rank > 8 heap spill count
distinct FusionTreeKey count
distinct FusionTreeBlockKey count
TreePairRowMemo hit/miss
persistent cache hit/miss
χ=32 energy parity vs TK
```

追加で見るべきものはこれです。

```text
row entry count per transform
terms before/after reduction
HashMap insertion count
same key accumulation count
parallel compile reproducibility
```

特に重要なのは、

```text
unique key が少なく clone が多いのか
unique key 自体が多いのか
```

です。

前者なら `TreeId` 化がかなり効きます。  
後者なら enumeration / block algorithm 側も見直す必要があります。

---

## Phase 1: 型を先に分ける

この段階では挙動を変えません。

追加する型はこれです。

```rust
pub struct VertexId(u32);

pub struct ExternalSpaceSpec {
    pub uncoupled: Box<[SectorId]>,
    pub is_dual: Box<[bool]>,
}

pub enum FusionPathSpec {
    Unique,
    Simple {
        innerlines: Box<[SectorId]>,
    },
    Generic {
        innerlines: Box<[SectorId]>,
        vertices: Box<[VertexId]>,
    },
}

pub struct FusionTreeSpec {
    pub external: ExternalSpaceSpec,
    pub coupled: SectorId,
    pub path: FusionPathSpec,
}
```

既存の `FusionTreeKey` には変換 API を足す。

```rust
impl FusionTreeKey {
    pub fn to_spec(&self, style: FusionStyleKind) -> Result<FusionTreeSpec, CoreError>;
    pub fn from_spec(spec: &FusionTreeSpec) -> Self;
}
```

この時点では `FusionTreeKey` の内部構造はまだ変えない。

目的は、

```text
semantic tree
runtime tree key
```

を概念上分けることです。

---

## Phase 2: `TreeArena` を入れる

次に category-scoped `TreeArena` を入れます。

```rust
impl TreeArena {
    pub fn intern_spec(&mut self, spec: &FusionTreeSpec) -> TreeId;
    pub fn resolve(&self, id: TreeId) -> FusionTreeSpecRef<'_>;
}
```

最初は lock-free な local arena だけでもよいです。

```rust
pub struct LocalTreeArena {
    map: FxHashMap<CompactTreeKey, LocalTreeId>,
    specs: Vec<FusionTreeSpec>,
}
```

ここでは、既存の transform function はまだ `FusionTreeKey` を返してよいです。  
返ってきた key をすぐ intern し、以後は `TreeId` を使う。

この段階で hot accumulator を変えられます。

---

## Phase 3: accumulator を `TreePairId` 化する

現在の accumulator は generic `K` を key にしています。  
この hot path 専用に `TreePairId` 版を作る。

```rust
pub enum TreePairTermAccumulator<S> {
    Empty,
    Singleton(TreePairId, S),
    Map {
        order: Vec<TreePairId>,
        coefficients: FxHashMap<TreePairId, S>,
    },
}
```

または term 数が多い block path では、

```rust
Vec<(TreePairId, S)>
```

へ push して sort/reduce する。

この Phase が、最初に大きく効く可能性が高いです。

この時点の goal は、

```text
compose_block_terms 内で FusionTreeBlockKey を HashMap key にしない
```

ことです。

---

## Phase 4: row memo key を `TreePairId` にする

次に `TreePairRowMemo` の key を変更します。

現在の概念は、

```rust
FxHashMap<(RuleKey, TreeTransformOperation, FusionTreeBlockKey), Arc<Rows>>
```

です。([raw.githubusercontent.com](https://raw.githubusercontent.com/Ryo-wtnb11/TeNeT/ac636fefb5864a1f6e5bf843cd419f958494da7e/tenet-tensors/src/tree_transform/plan.rs))

これを、

```rust
FxHashMap<(RuleKey, OperationId, TreePairId), Arc<Rows>>
```

にする。

ただし、この時点で persistent cache に `TreePairId` を保存してはいけません。  
disk には semantic spec を保存する。

メモリ上だけ `TreePairId` を使う。

---

## Phase 5: id-native recoupling に移行する

ここまでは、既存の `FusionTreeKey` を生成してから intern する方式です。  
これは中間段階としてはよいですが、best ではありません。

最終的には、tree operation 自体を id-native にします。

たとえば今は概念的に、

```rust
fn multiplicity_free_artin_braid_at_with_inverse(
    rule: &R,
    tree: &FusionTreeKey,
    index: usize,
    inverse: bool,
) -> Vec<(FusionTreeKey, Scalar)>
```

のような形です。

最終形は、

```rust
fn artin_braid_at_with_inverse(
    ctx: &mut RecouplingContext<R>,
    tree: TreeId,
    index: usize,
    inverse: bool,
) -> Vec<(TreeId, Scalar)>
```

にする。

tree の中身が必要なときだけ arena から slice view を取る。

```rust
let view = ctx.tree_arena.view(tree);
let uncoupled = view.uncoupled();
let innerlines = view.innerlines();
let vertices = view.vertices();
```

この Phase で初めて、`FusionTreeKey` construction/clone が hot path から消えます。

---

## Phase 6: CompactKey を入れる

`TreeArena` の lookup key に compact exact key を入れます。

```rust
pub enum CompactExternalSpaceKey {
    Inline { bytes: [u8; 64], len: u8 },
    Heap(Box<[u8]>),
}

pub enum CompactPathKey {
    Inline { words: [u64; 4], len_inner: u8, len_vertex: u8 },
    Heap(Box<[u8]>),
}
```

encoding は exact にする。

推奨 encoding は、

```text
rank
style tag
sector id sequence
dual bitset
coupled sector
innerline sequence
vertex sequence
```

です。

SectorId が小さい場合は varint encoding で詰める。  
ただし `SectorId` が大きくても exact に encode できる必要があります。

この Phase は optimization です。  
`TreeId` 化より優先度は低いです。

---

## Phase 7: block-local dense index を入れる

block transform では、さらに dense id を使います。

```rust
pub struct BlockTreeBasis {
    pub external: ExternalSpaceId,
    pub trees: Vec<TreeId>,
    pub dense_index: FxHashMap<TreeId, DenseTreeIndex>,
}
```

tree pair なら、

```rust
pub struct BlockTreePairBasis {
    pub codomain_external: ExternalSpaceId,
    pub domain_external: ExternalSpaceId,

    pub pairs: Vec<TreePairId>,
    pub dense_index: FxHashMap<TreePairId, DensePairIndex>,
}
```

ここまで行くと、row assembly は

```rust
Vec<Option<RowId>>
Vec<Scalar>
```

のような dense structure に寄せられます。

つまり HashMap は outer memo lookup だけに残し、inner compose から消していく。

---

## Phase 8: Generic fusion を有効化する

Generic fusion support を入れるなら、この時点でやるのが自然です。

追加すべき trait surface は概念的にはこれです。

```rust
pub trait GenericFusionRule: FusionRule {
    fn fusion_multiplicity(&self, a: SectorId, b: SectorId, c: SectorId) -> usize;

    fn f_symbol(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        // old path labels
        old_inner: SectorId,
        old_vertices: &[VertexId],
        // new path labels
        new_inner: SectorId,
        new_vertices: &[VertexId],
    ) -> Self::Scalar;

    fn r_symbol(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        mu: VertexId,
        nu: VertexId,
    ) -> Self::Scalar;
}
```

実際の signature はもっと整理できますが、重要なのは、

```text
vertices が F/R-symbol の matrix indices になる
```

という点です。

この Phase では、multiplicity-free の path は `Simple` のまま、Generic だけ `vertices` を明示的に持つ。

---

## Phase 9: persistent cache format を更新する

cache version を上げます。

保存するのは id ではなく semantic data です。

```text
TREE_PLAN_CACHE_VERSION = 2

category fingerprint
operation spec
external spaces
tree spaces
paths
row entries
```

load 時は、

```text
validate fingerprint
semantic tree/path を arena に intern
TreeId を再生成
rows を復元
```

する。

この Phase では、old v1 cache を捨ててもよいです。  
persistent cache は correctness ではなく optimization なので、互換性を無理に保つ必要はありません。

---

## Phase 10: `FusionTreeKey` を semantic wrapper に落とす

最後に、`FusionTreeKey` の役割を縮小します。

```text
Before:
    FusionTreeKey = semantic key + runtime key + cache key

After:
    FusionTreeKey = semantic/debug/serialization compatibility type
    TreeId        = runtime key
```

この段階で、`FusionTreeKey` の内部を変えるかどうかを決めます。

選択肢は 2 つです。

### Option A: legacy compatibility として残す

```rust
pub struct FusionTreeKey {
    uncoupled: SectorVec,
    coupled: Option<SectorId>,
    is_dual: DualVec,
    innerlines: SectorVec,
    vertices: SectorVec,
}
```

これは既存コードとの互換性が高い。

### Option B: `FusionTreeSpec` wrapper にする

```rust
pub struct FusionTreeKey {
    spec: FusionTreeSpec,
}
```

将来設計としては B がきれいです。  
ただし migration cost は高いです。

---

# 15. 検証 gate

この変更は blast radius が大きいので、merge gate は厳しくするべきです。

最低限：

```text
χ=32 energy matches TK to 10 digits or better
existing tree transform parity tests pass
row count parity
block structure parity
cold time no regression
warm time no regression
allocation count decreases
persistent cache replay correctness
parallel compile reproducibility
```

追加で、id 化後はこれも見るべきです。

```text
same input repeated 100 times -> identical row order
same input with recoupling_threads=1 and >1 -> identical plan output
same cache loaded from disk -> identical plan output
```

floating-point bit 一致を目標にするなら、特に

```text
parallel id assignment
HashMap iteration order
coefficient reduction order
```

を固定する必要があります。

---

# 16. 着手判断

現在の前提だけなら、

```text
warm はすでに良い
cold は persistent cache で償却可能
FusionTreeKey clone/hash/eq は cold の一部
```

なので、純粋な短期 performance 目的で全 refactor するのは割に合わない可能性があります。

しかし、将来 Generic fusion / multiplicity ありまで行くなら話が変わります。

その場合、

```text
vertices を identity に含める
F/R-symbol が matrix 化する
row が sparse matrix 的になる
cache key が category/gauge に依存する
```

ので、`FusionTreeKey` をそのまま runtime key にし続ける設計はかなり苦しくなります。

したがって判断はこうです。

| 目的 | 判断 |
|---|---|
| 現在の multiplicity-free cold を数 % 改善したいだけ | full refactor は過剰 |
| rank > 8 spill を少し減らしたい | SmallVec capacity experiment で十分 |
| Generic fusion まで行く | arena + TreeId 化を設計軸にすべき |
| long-term Rust-native TN core を作る | semantic tree と runtime id を分離すべき |

あなたのプロジェクト方針なら、**TreeId arena 設計を最終目標に置くのが正しい**です。

---

# 17. 最終的な優先順位

最も重要な順に並べるとこうです。

1. **semantic tree と runtime id を分ける**
2. **`VertexId` を `SectorId` から分ける**
3. **Generic fusion では `vertices` を path identity に含める**
4. **`ExternalSpace = uncoupled + is_dual` を共有する**
5. **`TreeSpace = ExternalSpace + coupled` にする**
6. **`PathArena` を style-dependent にする**
7. **hot path を `TreeId` / `TreePairId` にする**
8. **block-local dense index を使う**
9. **CompactKey は arena lookup 専用に使う**
10. **persistent cache は semantic data を保存する**
11. **deterministic order を強制する**
12. **固定 inline buffer や cached hash は補助扱いにする**

---

# 18. 一文でまとめる

TeNeT の best design は、

```text
FusionTreeKey を速い値型にすること
```

ではなく、

```text
FusionTreeKey を semantic description に降格し、
category-scoped TreeArena に canonicalize し、
recoupling / cache / accumulator では TreeId と TreePairId だけを使うこと
```

です。

そして multiplicity ありまで考えるなら、

```text
PathId は innerlines だけでなく vertices も含む
```

必要があります。

これが、動的 rank を維持しつつ、Rust らしく、Generic fusion まで伸ばせる設計です。
