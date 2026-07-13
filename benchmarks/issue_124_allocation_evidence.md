# Issue 124 allocation evidence

## Measurement boundary

`tenet-network/tests/intermediate_allocations.rs` runs every `(workload, chi,
mode)` in an isolated process. Both modes warm the same execution-context and
slot paths. The fresh-intermediate mode clears only arena payloads before a
probe; the reuse mode retains them.

The allocator records alloc, realloc, and dealloc calls and bytes. A fixed,
allocation-free pointer registry tracks allocations originating inside the
probe, which makes current and peak live bytes independent of delayed frees for
objects allocated during warm-up. Three probes are reported; hard gates use the
median so one delayed backend-worker allocation cannot select the result.

Payload allocations are classified by the oracle output storage size. These
matrix-chain fixtures have equal-sized contraction, orientation, and final
payloads. The gate therefore distinguishes:

- payload alloc calls and peak payload bytes;
- payload bytes retained by non-final intermediates after the output is freed;
- bytes held by the escaping final output before it is freed;
- all remaining metadata and backend scratch, retained as diagnostics; and
- executor-owned contraction/orientation/final-output event counters.

Size classification is deliberately fixture-local. It does not claim that an
arbitrary network has equal-sized intermediates, nor that a same-sized backend
allocation can never exist. Structural executor counters independently prove
which contraction and orientation paths were reused.

The matrix covers `chi = 8, 16, 32, 64`, real and complex storage, multiblock
SU(2) and SU(3), and a schedule with a nonidentity intermediate orientation.
The Rust global allocator cannot observe CUDA device allocations.

## Final escaping output

The remaining payload allocation is the `Tensor` returned to the caller. It
cannot be placed back in the workspace arena because its `Arc<Data>` escapes
the execution lifetime.

A separate API PR should add a host-only first slice:

```text
PlannedNetwork::execute_overwrite_into(tensors, destination, workspace)
    -> Result<(), Error>
```

It should validate runtime identity, rule, dtype, exact fusion layout, unique
destination ownership, and input non-aliasing before mutation. The final
contraction or orientation should write directly into the caller destination
with the existing beta-zero overwrite contract. The owned-return API remains a
convenience wrapper. Required tests are NaN-seeded destination oracles, every
pre-mutation rejection, held prior outputs, and a measured zero final-payload
allocation after warm-up.

## CUDA arena reuse

The current arena is host-only. CUDA storage does not expose the stream/event
identity needed to prove that a returned buffer is no longer in flight. A
separate CUDA PR should:

1. bind each arena to one CUDA execution context and stream identity;
2. record a completion event when returning a buffer;
3. wait on that event before reuse from another stream or worker;
4. instrument device allocations separately from Rust host allocations; and
5. verify that measured replay introduces neither device allocation nor a
   device-to-host synchronization.

Synchronizing on every arena return would be correct but would erase the
intended overlap, so it is not used as an implicit fallback in this evidence
slice.
