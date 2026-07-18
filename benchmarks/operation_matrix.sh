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

# Keep plain variables: the macOS system bash is 3.2 and has no associative
# arrays.  This is a diagnostic script, so portability is more useful than a
# dynamic map here.
user_api_bin=$(build_test_binary user_api)
user_decomp_bin=$(build_test_binary user_decomp)

echo "# build complete; timings below exclude compilation (binary startup remains)"

for spec in \
  "permute user_api permute_roundtrip_restores_the_tensor" \
  "transpose user_api transpose_and_adjoint_involutions" \
  "svd_compact user_decomp svd_compact_reconstructs_u1_and_su2" \
  "qr_compact user_decomp qr_and_lq_factorizations" \
  "eigh_full user_decomp eigh_reconstructs_a_hermitized_tensor"; do
  read -r op test filter <<<"$spec"
  case "$test" in
    user_api) binary=$user_api_bin ;;
    user_decomp) binary=$user_decomp_bin ;;
    *) echo "unknown test binary: $test" >&2; exit 1 ;;
  esac
  echo "## $op (fresh-process cold/repeat; startup + operation, nanoseconds)"
  python3 - "$binary" "$filter" "$REPS" <<'PY'
import statistics
import subprocess
import sys
import time

binary, test_filter, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
def run_once():
    start = time.perf_counter_ns()
    subprocess.run([binary, test_filter, "--exact"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter_ns() - start

cold = run_once()
repeats = [run_once() for _ in range(reps)]
print(f"cold_ns={cold}")
print(f"repeat_ns={repeats}")
print(f"repeat_median_ns={statistics.median(repeats)}")
print("note=samples are separate processes; same-process warm timing is unavailable from Rust test binaries")
PY
done
