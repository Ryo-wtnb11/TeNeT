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

- The **traits exist**, and the dense linear-algebra seam is **selectable at
  the builder** two ways (#64):
  - `Runtime::builder().linalg_backend(LinalgBackend::Faer | LinalgBackend::Blas)`
    picks a shipped provider. Default is faer. `Blas` uses the system
    BLAS/LAPACK linked via a `blas-*` cargo feature and fails at `build()` if
    none was compiled in.
  - `Runtime::builder().with_dense_executor(Box<dyn DenseExecutor + Send>)`
    injects a custom backend (takes precedence over `linalg_backend`). The
    runtime holds it behind `Box<dyn DenseExecutor>`; unset uses the
    faer-backed `DefaultDenseExecutor` (`tenet-dense` → `tenferro` → faer/gemm).
  - Runtime vs compile-time: OpenBLAS / MKL / Accelerate can't be linked
    simultaneously, so *which* BLAS is a compile-time `blas-*` feature; at
    runtime you choose faer vs the one linked BLAS.
- Still hardcoded: the transpose/contract seam is fixed to
  `DenseTreeTransformOperations`.
- Device selection is exposed the same way (`Runtime::builder().cuda`).
- Remaining alternatives: HPTT (fast transpose) and TBLIS (transpose-free
  contraction) for the transform/contract seams — see #7, #41.

Remaining work: expose the transform/contract seam selection at the builder the
same way the linear-algebra provider now is.

## Rules

1. **Everything behind a trait.** No compute primitive is inlined into operator
   or user-layer code. Adding a backend is a new trait `impl` plus registration
   — it must require *zero* changes to operators (`contract`, `adjoint`, `svd`,
   …).

2. **Selection is explicit and runtime, at `Runtime::builder()`.** Device
   selection already works this way (`Runtime::builder().cuda(device)`). CPU
   compute backends must be selectable the same way — the dense linear-algebra
   provider (`.linalg_backend(LinalgBackend::Faer | LinalgBackend::Blas)`, done
   in #64) and, still to come, the transpose/contract kernels
   (`.transpose_backend(...)`, `.contract_backend(...)`) — not chosen at call
   sites and not hardcoded at compile time. There is one documented default
   (faer / CPU dense); everything else is opt-in. (Where several
   implementations of one provider can't co-link — OpenBLAS vs MKL — which one
   is a compile-time feature; the *family* stays a runtime choice.)

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

## Adding a backend (checklist)

- Implement the relevant backend trait; keep it `tenferro`-side if it is a heavy
  kernel, exposed to `tenet` via an adapter.
- Wire a selection knob on `RuntimeBuilder`; keep the default unchanged.
- Prove numerical equivalence against the existing suite (no result changes).
- Gate adoption on a benchmark that shows the win in the intended regime, and
  keep that metric as a merge gate.
