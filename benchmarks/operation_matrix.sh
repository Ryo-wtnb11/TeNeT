#!/usr/bin/env bash
# How: run the already-built public TeNeT paths with a fixed single-thread
# environment and report cold (new process) versus warm (repeated process)
# timings.  This script deliberately does not alter caches or production code.
# What: owned factorization/permute paths are measured with cargo test filters;
# low-level destination permute is listed separately because it needs a typed
# TensorMap destination constructed by the caller.
# Why: TensorKit's reference suite benchmarks `permute!`/`svd_compact!` into
# caller-owned storage; mixing those with TeNeT's owned-returning API hides the
# actual algorithmic cost behind output allocation.

set -euo pipefail
cd "$(dirname "$0")/.."

: "${RAYON_NUM_THREADS:=1}"
: "${REPS:=5}"
: "${CARGO_TARGET_DIR:=target/operation-matrix}"
export RAYON_NUM_THREADS CARGO_TARGET_DIR

echo "operation,mode,command"
echo "permute,owned,cargo test --release -p tenet --test user_api permute_roundtrip_restores_the_tensor"
echo "transpose,owned,cargo test --release -p tenet --test user_api transpose_and_adjoint_involutions"
echo "svd_compact,owned,cargo test --release -p tenet --test user_decomp svd_compact_reconstructs_u1_and_su2"
echo "qr_compact,owned,cargo test --release -p tenet --test user_decomp qr_and_lq_factorizations"
echo "eigh_full,owned,cargo test --release -p tenet --test user_decomp eigh_reconstructs_a_hermitized_tensor"
echo "permute,destination,tenet_tensors::permute_into (typed TensorMap destination; see facade API)"
echo "transpose,destination,tenet_tensors::transpose_into (typed TensorMap destination; see facade API)"
echo

build_test_binary() {
  local test_crate=$1
  # Build once.  All subsequent samples execute the binary directly, so cargo
  # dependency/build time cannot be mistaken for operation time.
  cargo test --release -p tenet --test "$test_crate" --no-run >/tmp/tenet-op-matrix-build.out
  local binary
  binary=$(find "$CARGO_TARGET_DIR/release/deps" -type f -perm -111 \
    -name "${test_crate}-*" ! -name '*.d' | head -n 1)
  test -n "$binary" || { echo "missing test binary for $test_crate" >&2; return 1; }
  printf '%s\n' "$binary"
}

declare -A BIN
for test_crate in user_api user_decomp; do
  BIN[$test_crate]=$(build_test_binary "$test_crate")
done

echo "# build complete; timings below exclude compilation (binary startup remains)"

for spec in \
  "permute user_api permute_roundtrip_restores_the_tensor" \
  "transpose user_api transpose_and_adjoint_involutions" \
  "svd_compact user_decomp svd_compact_reconstructs_u1_and_su2" \
  "qr_compact user_decomp qr_and_lq_factorizations" \
  "eigh_full user_decomp eigh_reconstructs_a_hermitized_tensor"; do
  read -r op test filter <<<"$spec"
  binary=${BIN[$test]}
  echo "## $op (cold, fresh test-binary process; startup + operation, seconds)"
  /usr/bin/time -p "$binary" "$filter" --exact --nocapture 2>&1 | tail -n 8
  echo "## $op (warm, $REPS repeated test-binary processes; startup + operation, seconds)"
  for ((i=1; i<=REPS; i++)); do
    /usr/bin/time -p "$binary" "$filter" --exact >/tmp/tenet-op-matrix.out 2>/tmp/tenet-op-matrix.time
    printf '%s ' "$i"
    awk '/^real / {print $2}' /tmp/tenet-op-matrix.time
  done
done
