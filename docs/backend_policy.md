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
    `linalg_backend`). An unset built-in kind follows Tenferro's resolved
    compiled provider default: BLAS when its CPU build enables `cpu-blas`,
    otherwise faer.
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

Device operations take only a process-wide lock per CUDA device ordinal, the
runtime's own CUDA-context mutex, and Tenferro's internal handle and plan
locks, not the Runtime state mutex, so Host work on the same runtime is not
blocked by device work. This is not a concurrency, overlap, or multi-stream
claim: device operations of every Runtime on one device serialize their
host-side enqueue on that device lock (GPU execution stays asynchronous; the
lock adds no host synchronization). A host synchronization that already
happens under a lease — `to_host`, scalar and spectrum downloads in
reductions and factorizations —
now stalls every Runtime on the device, not only the caller.

The device lock is shared across Runtimes because CubeCL's client is
process-wide per device and publishes a binding's stream cursor only when it
is bound, not when a later kernel writes it (tensor4all/cubecl#16). Without
it, a second Runtime could sync past an output's bind before its write and
then read it unfinished (#1384). The lock covers outputs bound and fully
written within one lease, which is every operation that returns a new tensor.

A buffer bound in an earlier lease and written again later — a
`*_overwrite_into` destination, or reused scratch (`CudaContractScratch`,
pooled `tensor!` network intermediates) — spans two leases, so the lock alone
cannot order it: another thread that synced past its bind in between would
read the later write unfinished, or overwrite it while an earlier read is
still in flight (#1391; reproduced on an A100 for overwrite destinations).
Opening the first CUDA context of the process (`CudaDenseContext::new`)
therefore sets CubeCL's process-wide `streaming.max_streams` to 1, taking
`cubecl.toml` and the environment as CubeCL would and overriding only that
field. Every CubeCL command and every Tenferro vendor call (cuBLAS, cuSOLVER,
cuTENSOR use the same stream slot) of every thread then runs on one CUDA
stream per device, in enqueue order, which follows happens-before between
threads; the unpublished cursor is never needed for ordering. This also
covers CubeCL or Tenferro users of the device outside TeNeT in the same
process, provided TeNeT opens its device first. If something else loaded
CubeCL's configuration first with more than one stream, opening a device
fails with `DenseError::Unsupported` rather than running unordered.

Enqueue order follows happens-before because CubeCL's server queue is FIFO
and a vendor call drains it first: raw sessions, memsets and scalar downloads
call `flush_cubecl`, while cuTENSOR and cuBLAS rely on the blocking
`get_resource` in Tenferro's `typed_device_ptr`. That function skips the
blocking call for a buffer created on the calling thread whose address it
has cached, so a CubeCL kernel still queued unflushed into such a buffer can
reach the stream after a later vendor call on it (tensor4all/tenferro-rs#1868;
independent of the stream count, and possible within one thread). In TeNeT
only the empty-contraction scale or fill of an owned destination queues such
a kernel.

Process-wide side effects: a `cubecl.toml` `streaming.max_streams` value is
overridden without notice; the setting stays fixed even when opening the
device then fails (for example without a GPU); every other CubeCL client in
the process, including wgpu, also gets one stream; and a later
`CubeClRuntimeConfig::set` panics, since CubeCL allows setting it only once.

Cost: GPU work of different threads no longer overlaps on the device (it
serializes on the stream, not only at enqueue), and Tenferro's cross-thread
host `synchronize()` no longer happens because every thread shares one slot.

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
