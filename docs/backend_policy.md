# Backend selection policy

TeNeT's design axis is maintainability + extensibility (idiomatic Rust) *and*
speed on dynamic-rank tensor networks. A first-class part of that extensibility
is being able to **choose the execution backend** for a compute primitive
(linear algebra, transpose/tree-transform, contraction/GEMM, device placement)
rather than hardcoding a single implementation. This document is the rule for
how backends are structured and selected.

> **See also** [complexity_parity_policy.md](complexity_parity_policy.md): a
> backend may differ from TensorKit in mechanism and constant factor, but never
> in asymptotic FLOP/storage order.

## Guiding principle

**If a seam is a plausible backend choice, abstract it as a backend from the
start** — put it behind a trait and make it selectable at `Runtime::builder()`
from day one. Do not hardcode one implementation and retrofit selection later.
Linear algebra is the canonical example: on CPU there are several viable
providers (faer, system BLAS/LAPACK, Intel MKL, OpenBLAS), so it must be a
selectable backend, not a fixed dependency — the same way device placement
(CPU vs CUDA) already is.

## What "backend" means here

A backend is a swappable implementation of a compute primitive, living behind a
trait:

- `DenseExecutor` — the dense linear algebra (per-coupled-sector GEMM, SVD, eig,
  QR, inv, exp). `tenet-matrixalgebra` is already generic over it
  (`fn eigh_full_dyn<E>(dense: &mut E, …)`).
- `TreeTransformBackend` — tree transforms / transposes (permute, braid, twist).
- `TensorContractBackend` / `HostTensorContractBackend` — contraction (the
  per-coupled-sector GEMMs).
- `TensorTraceOperationsBackend` — traces.
- Device placement — CPU vs CUDA, selected on the runtime.

State today, and the gap this policy targets:

- The **traits exist**, and the CPU dense provider is **selectable at the
  builder** with two independent knobs, both defaulting to faer (#64):
  - `Runtime::builder().linalg_backend(LinalgBackend::Faer | LinalgBackend::Blas)`
    picks the provider for the **factorizations** (SVD / QR / eigh / eig / inv /
    exp — LAPACK-style work).
  - `Runtime::builder().gemm_backend(LinalgBackend::Faer | LinalgBackend::Blas)`
    picks the provider for the **contraction GEMM** (`compose` / `contract` and
    recoupling replays — BLAS-style work). Independent of `linalg_backend`.
  - `Runtime::builder().with_dense_executor(Box<dyn DenseExecutor + Send>)`
    injects a custom factorization backend (takes precedence over
    `linalg_backend`). Unset defaults use the faer-backed
    `DefaultDenseExecutor` (`tenet-dense` → `tenferro` → faer).
  - `Blas` uses the system BLAS/LAPACK linked via a `blas-*` cargo feature and
    fails at `build()` if none was compiled in. Runtime vs compile-time:
    OpenBLAS / MKL / Accelerate can't be linked simultaneously, so *which* BLAS
    is a compile-time `blas-*` feature; at runtime you choose faer vs the one
    linked BLAS. (MKL, being both BLAS and LAPACK, backs both knobs at once.)
- The **transpose kernel** for pure permuted copies is selectable (#114):
  - `Runtime::builder().transpose_backend(TransposeBackend::FusedLoops | TransposeBackend::StridedPerm)`
    picks the kernel for the pack / assign-scatter copies of tree-transform
    replay and fusion-block contraction. Default `FusedLoops` (the zero-alloc
    fused loop nest) — byte- and dispatch-identical to before the knob existed.
    See [Transpose backend](#transpose-backend-fusedloops-vs-stridedperm) below
    for the measured regimes; `StridedPerm` is opt-in on those numbers.
- Still hardcoded: the transpose-free contraction path inside
  `DenseTreeTransformOperations` — TBLIS-style kernels are not yet a builder
  choice.
- Device selection is exposed the same way (`Runtime::builder().cuda`).

Remaining work: a true HPTT binding as a further `TransposeBackend` variant and
TBLIS (transpose-free contraction) for the contract kernels — see #7, #41.

## Transpose backend: FusedLoops vs StridedPerm

`TransposeBackend::StridedPerm` routes non-scaling, non-conjugating permuted
copies with a genuine stride-1 axis on both sides (after dropping extent-1
axes) through `strided_perm`'s HPTT-style blocked transpose
(`strided_kernel::copy_into_col_major`); everything else falls back to the
fused loop. Routed copies are **byte-identical** to the fused loop (checksum-
verified across the whole A/B grid), so this is purely a performance knob.

Measured regimes (#114 A/B: `microbench_fusion`, blas-accelerate,
`RAYON_NUM_THREADS=1`, median of 3; Δ% = StridedPerm vs FusedLoops, noise
floor ±3% from the transpose-free `compose` control):

| Regime | Δ% | Verdict |
|---|---|---|
| SU(2) d=4 swap / swap+out | **+92 / +95** | catastrophic loss |
| SU(2) d=8 swap / swap+out | +7 / +14 | loss |
| abelian (U1 / fZ2 / U1×fZ2), d≤8 | −2…+2 | neutral (noise) |
| d=16, all symmetries | −3…+1 | neutral, except ↓ |
| fZ2 swap+out d=16 | **−6.5** | the one clear win |

Profile of the loss (SU(2) d=4 swap): tree-replay pack ×6.8 and scatter ×6.1 —
`strided_perm`'s **per-call plan build** cannot amortize over the many tiny
blocks of small-degeneracy SU(2) replay, while the fused loop is zero-alloc
per call.

**Guidance:** keep the default. Consider `StridedPerm` only for large-block
abelian transpose-heavy workloads (large degeneracy, few sectors, permutation-
dominated), and adopt it on a measurement of *your* workload, not this table.

**Re-evaluation trigger:** an upstream strided-perm compiled-plan /
plan-once-execute-many API (the strided-rs compiled-plan gap) would remove the
per-call plan-build cost behind the pack ×6.8 / scatter ×6.1 blow-up, which is
the entire small-block loss; re-run the #114 A/B grid when it lands before
reconsidering the default.

## Rules

1. **Everything behind a trait.** No compute primitive is inlined into operator
   or user-layer code. Adding a backend is a new trait `impl` plus registration
   — it must require *zero* changes to operators (`contract`, `adjoint`, `svd`,
   …).

2. **Selection is explicit and runtime, at `Runtime::builder()`.** Device
   selection already works this way (`Runtime::builder().cuda(device)`). CPU
   compute backends must be selectable the same way — the dense factorization
   provider (`.linalg_backend(...)`) and the contraction GEMM provider
   (`.gemm_backend(...)`), both done in #64, the transpose kernel
   (`.transpose_backend(...)`, done in #114), and, still to come, the
   transpose-free-contraction kernels (TBLIS) — not chosen at call sites and
   not hardcoded at compile time.
   There is one documented default (faer / CPU dense); everything else is
   opt-in. (Where several implementations of one provider can't co-link —
   OpenBLAS vs MKL — which one is a compile-time feature; the *family* stays a
   runtime choice.)

3. **Separate WHAT from WHICH.** Operator and user-layer code express *what* to
   compute — spaces, axes, conjugate flags, output order — and never *which*
   kernel runs it. Kernel/route choice belongs to the backend/selection layer.
   Example: the adjoint contraction fold hands the seam semantic flags
   (`conjugate=true`, remapped axes); whether that becomes a BLAS `op='C'`, an
   HPTT transpose, or a TBLIS call is the backend's business.

4. **No ad-hoc kernels in operator code.** No raw BLAS calls and no
   threshold-gated hand-written kernels sitting in operators. Only structural
   improvements (dispatch hoisting, batched-GEMM seams) or a new backend behind
   the trait.

5. **Backend choice is a performance knob, never a semantics knob.** Switching
   backends must not change numerical results — TeNeT stays TensorKit-equivalent
   regardless of backend. Every backend is verified against the same oracle /
   test suite, and a backend is adopted on measured wins (metric-gated), not on
   expectation.

6. **Routing in `tenet`, heavy kernels in `tenferro` + adapters.** 2D
   normalization and the direct/SVD-invariant routing decisions live in
   `tenet`; the heavy transpose/contract kernels (HPTT, TBLIS) sit behind
   adapters in `tenferro`. `tenet` routes; it does not embed kernels.

## Parallel execution (current state)

Ops on a shared `Runtime` scale with outer threads (#155, #176). Nothing holds
the coarse state mutex for a whole computation any more; each op leases its
execution machinery for its own duration and runs lock-free.

- **Standalone ops** (`contract`, `permute`, factorizations) lease from two
  pools on `RuntimeInner`:
  - `ContextPool` — per-rule `TensorExecutionContext`s (the `Ctxs` the locked
    state used to carry) for contract/permute/transpose and the polar/exp/inv/
    pinv factorizations.
  - `ExecutorPool` — `Box<dyn DenseExecutor + Send>` for SVD/QR/eigh/eig/null.
  Both mirror the network `WorkspacePool`: mint-on-empty, idle cap = one warm
  resource per core (`available_parallelism`), quarantine-on-panic (a resource
  live during a panic is dropped, not returned). Single-threaded use reuses one
  pooled resource and warms its caches exactly as the state did, so it stays
  byte-identical to the pre-#176 locked path.
- **Plan cache** lives behind its own mutex (`PlanCacheHome`), separate from the
  state mutex, so the `tensor!` network path never contends with standalone ops.
  A warm hit costs **one** plan-cache acquisition (config read + slot access
  folded into `with_plan_cache`) and **one** topology hash (residency check +
  LRU touch folded into a single `LruCache::get`).
- Each `Runtime` owns **one** `SharedCpuContext` (a single rayon pool); every
  minted context/executor and all transform backends bind to it, so `dense_
  threads`/rayon width is one process-level knob, not per-lease.
- **Escape hatch:** a custom executor injected via `with_dense_executor` is not
  mintable, so those runtimes fall back to the `state` lock for factorizations
  (context leasing still applies). This is the one path that still serializes.

Measured scaling (`examples/thread_scaling.rs`, warm rank-4 SU(2) `contract`,
`dense_threads(1)` + `RAYON_NUM_THREADS=1` so outer threads are the only
parallelism; speedup vs N=1):

| Arm | N=4 | N=8 |
|---|---|---|
| standalone shared `Runtime` | 3.84× | 7.58× |
| network cached-plan path | 3.77× | — |
| per-thread `Runtime` (ceiling) | ~3.9× | ~7.7× |

Standalone now tracks the per-thread ceiling. The residual gap at d=16, N=8 is
memory bandwidth (larger blocks), not lock contention. Data-parallel callers no
longer need a `Runtime` per thread — a shared handle scales — though one per
thread remains valid and contention-free.

## Adding a backend (checklist)

- Implement the relevant backend trait; keep it `tenferro`-side if it is a heavy
  kernel, exposed to `tenet` via an adapter.
- Wire a selection knob on `RuntimeBuilder`; keep the default unchanged.
- Prove numerical equivalence against the existing suite (no result changes).
- Gate adoption on a benchmark that shows the win in the intended regime, and
  keep that metric as a merge gate.
