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

## Validate-once replay

The checked arena dispatch retains the expected fusion layout after its first
reusable write. Every later write still validates runtime, rule, dtype,
placement, aliasing, unique destination ownership, operation axes, and the
exact destination layout before mutation. It does not rebuild the expected
layout to perform that validation.

`NetworkExecutionStats` counts contract and orientation layout preparations.
The release allocator worker warms both caches before its three measured
replays and rejects any nonzero preparation or structural-space-comparison
delta. The same worker therefore provides deterministic construction and
constant-time identity gates alongside allocator evidence.

The common replay compares interned `Arc` identities. A semantically equal
space arriving through a distinct `Arc` takes one structural fallback, then
the cache adopts that current identity. Tests cover every production facade
rule: U(1), Z2, fermion parity, SU(2), U(1) x fermion parity, fermion parity x
U(1) x SU(2), and the separate generic SU(3) route. The borrowed operation
path remains generic over `MultiplicityFreeRigidSymbols`; generic outer-
multiplicity cache completion remains outside this slice.

A rank-nine U(1) permutation exceeds the inline `AxisVec` capacity. After one
prepared write, its measured replay performs zero allocator and reallocator
calls, proving that the cached operation is borrowed through the
multiplicity-free transform cache rather than cloned into a heap-backed key.

On the issue-124 `chi = 16` fixtures, repeatable steady probes after backend
warm-up changed as follows:

| workload | before calls | after calls | before bytes | after bytes |
| --- | ---: | ---: | ---: | ---: |
| U(1) f64 contraction | 65 | 49 | 7372 | 7172 |
| SU(2) f64 intermediate orientation | 43 | 35 | 6988 | 6888 |

The output payload remains one allocation of 6144 bytes in every row. The
reduction is metadata work removed from the warm replay, not payload reuse
being reclassified.

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
