# Backend selection policy

TeNeT's design axis is maintainability + extensibility (idiomatic Rust) *and*
speed on dynamic-rank tensor networks. A first-class part of that extensibility
is being able to **choose the execution backend** for a compute primitive
(transpose/tree-transform, contraction/GEMM, device placement) rather than
hardcoding a single implementation. This document is the rule for how backends
are structured and selected.

## What "backend" means here

A backend is a swappable implementation of a compute primitive, living behind a
trait:

- `TreeTransformBackend` — tree transforms / transposes (permute, braid, twist).
- `TensorContractBackend` / `HostTensorContractBackend` — contraction (the
  per-coupled-sector GEMMs).
- `TensorTraceOperationsBackend` — traces.
- Device placement — CPU vs CUDA, selected on the runtime.

Concrete backends today: `DenseTreeTransformOperations` (CPU dense, the
default), `HostTensorOperations`, and the CUDA path. Planned: HPTT (fast
transpose) and TBLIS (transpose-free contraction) — see #7, #41.

## Rules

1. **Everything behind a trait.** No compute primitive is inlined into operator
   or user-layer code. Adding a backend is a new trait `impl` plus registration
   — it must require *zero* changes to operators (`contract`, `adjoint`, `svd`,
   …).

2. **Selection is explicit and runtime, at `Runtime::builder()`.** Device
   selection already works this way (`Runtime::builder().cuda(device)`). CPU
   compute backends must be selectable the same way (e.g.
   `.transpose_backend(...)`, `.contract_backend(...)`), not chosen at call
   sites and not hardcoded at compile time. There is one documented default
   (CPU dense); everything else is opt-in.

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
