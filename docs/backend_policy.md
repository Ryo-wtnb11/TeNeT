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

**Start at the supported Tenferro/Strided boundary.** Add a trait, builder
knob, or retained execution state only when a concrete supported requirement
cannot be expressed through that boundary. TeNeT owns categorical semantics
and layout decisions; Tenferro owns dense-provider resources and Strided owns
applicable strided movement. Do not pre-abstract every plausible seam.

## What "backend" means here

Existing execution seams include:

- `DenseExecutor` — the dense linear algebra (per-coupled-sector GEMM, SVD, eig,
  QR, inv, exp). `tenet-matrixalgebra` is already generic over it
  (`fn eigh_full_dyn<E>(dense: &mut E, …)`).
- `TreeTransformBackend` — tree transforms / transposes (permute, braid, twist).
- `TensorContractBackend` / `HostTensorContractBackend` — contraction (the
  per-coupled-sector GEMMs).
- `TensorTraceOperationsBackend` — traces.
- Device placement — CPU vs CUDA, selected on the runtime.

State today:

- The CPU dense provider is selectable at the builder with two independent
  knobs:
  - `Runtime::builder().linalg_backend(LinalgBackend::Faer | LinalgBackend::Blas)`
    picks the provider for the **factorizations** (SVD / QR / eigh / eig / inv /
    exp — LAPACK-style work).
  - `Runtime::builder().gemm_backend(LinalgBackend::Faer | LinalgBackend::Blas)`
    picks the provider for the **contraction GEMM** (`compose` / `contract` and
    recoupling replays — BLAS-style work). Independent of `linalg_backend`.
  - `Runtime::builder().with_dense_executor(Box<dyn DenseExecutor + Send>)`
    injects a custom factorization backend (takes precedence over
    `linalg_backend`). An unset built-in kind follows the compiled provider
    default. Earlier documentation calling that default universally faer is a
    separate pending default-provider decision; this policy does not choose it.
  - `Blas` uses the system BLAS/LAPACK linked via a `blas-*` cargo feature and
    fails at `build()` if none was compiled in. Runtime vs compile-time:
    OpenBLAS / MKL / Accelerate can't be linked simultaneously, so *which* BLAS
    is a compile-time `blas-*` feature; at runtime you choose faer vs the one
    linked BLAS. (MKL, being both BLAS and LAPACK, backs both knobs at once.)
- Device selection is exposed the same way (`Runtime::builder().cuda`).

## Rules

1. **Keep operator code independent of dense providers.** Reuse an existing
   Tenferro/Strided boundary where it expresses the requirement. Introduce a
   new trait only for a concrete supported capability that cannot be represented
   there; operators (`contract`, `adjoint`, `svd`, …) do not select providers.

2. **Selection follows the supported boundary.** Device selection is explicit
   at `Runtime::builder().cuda(device)`. Built-in dense factorization and GEMM
   providers are selected with `.linalg_backend(...)` and `.gemm_backend(...)`;
   custom factorization executors can be injected. A new runtime knob requires
   a concrete capability, not merely a possible future implementation. Where
   OpenBLAS and MKL cannot co-link, the linked implementation remains a
   compile-time feature.

3. **Separate WHAT from WHICH.** Operator and user-layer code express *what* to
   compute — spaces, axes, conjugate flags, output order — and never *which*
   kernel runs it. Kernel/route choice belongs to the backend/selection layer.
   Example: the adjoint contraction fold hands the seam semantic flags
   (`conjugate=true`, remapped axes); the provider chooses its dense operation.

4. **No ad-hoc kernels in operator code.** No raw BLAS calls or
   threshold-gated hand-written numerical kernels in operators. Structural
   improvements route through the existing Tenferro/Strided boundary; a new
   backend needs a concrete unsupported requirement.

5. **Backend choice is a performance knob, never a semantics knob.** Supported
   mathematical semantics and dtype-appropriate numerical tolerances remain
   the same, while provider capabilities can differ explicitly. Correctness and
   capability gates are distinct from separately recorded performance evidence;
   wall-clock timing is not a CI pass/fail gate.

6. **Routing in `tenet`, dense kernels in `tenferro` + adapters.** 2D
   normalization and direct/SVD-invariant routing decisions live in `tenet`;
   dense kernels remain behind Tenferro adapters. `tenet` routes; it does not
   embed kernels.

## Parallel execution (current state)

Standalone operations lease contexts and mintable dense executors from runtime
pools. Checkout and return synchronize on those pools; the plan cache and
shared structural stores also synchronize independently. The pool cap is
`max(available_parallelism, 2)`, falling back to 4 when the system does not
report a value. A leased resource that unwinds is not returned to its pool.

Built-in executors using the compiled default CPU kind share one
`SharedCpuContext` per runtime. An explicitly requested nondefault kind uses a
private provider context, and an injected executor owns its configuration and
falls back to the runtime state lock for factorization. `CpuContext` worker
resources are distinct from the process-global Rayon configuration and from
provider-internal synchronization. Consequently this design makes no general
lock-free, byte-identical warm-path, or outer-thread scaling guarantee.

## Historical context

Issues #155 and #176 record earlier implementation work; they are not current
Tenferro 0.5 scaling evidence. Any scaling claim needs fresh, pinned provider,
revision, configuration, and raw measurement evidence.

## Adding a backend (checklist)

- Start with an existing Tenferro/Strided boundary; add a trait or selection
  knob only for a concrete supported requirement it cannot express.
- Keep a dense kernel `tenferro`-side and expose it to `tenet` through an
  adapter when a new backend is warranted.
- Prove supported mathematical semantics with dtype-appropriate tolerances and
  test unsupported provider capabilities explicitly.
- Record separately scoped performance evidence when relevant; do not use a
  wall-clock benchmark as a CI merge gate.
