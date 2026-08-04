#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RAYON_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
: "${CARGO_TARGET_DIR:=target/operation-matrix}"
: "${OP_MATRIX_GEMM_BACKEND:=faer}"
: "${OP_MATRIX_CARGO_FEATURES:=cpu-faer}"
export RAYON_NUM_THREADS OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS CARGO_TARGET_DIR OP_MATRIX_GEMM_BACKEND
export TENET_AUTHORITY="$(git describe --always --dirty)"
export TENFERRO_AUTHORITY="$(git -C ../tenferro-rs describe --always --dirty 2>/dev/null || printf unavailable)"

cpu_name="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
if [[ -z "$cpu_name" ]] && command -v system_profiler >/dev/null 2>&1; then
    cpu_name="$(system_profiler SPHardwareDataType | awk -F': ' '/Chip:/ { print $2; exit }')"
fi
if [[ -z "$cpu_name" && -r /proc/cpuinfo ]]; then
    cpu_name="$(awk -F: '/model name/ { sub(/^[[:space:]]+/, "", $2); print $2; exit }' /proc/cpuinfo)"
fi
: "${cpu_name:=unknown}"
printf '# host_os=%s host_release=%s host_arch=%s cpu=%s\n' \
    "$(uname -s)" "$(uname -r)" "$(uname -m)" "$cpu_name"
printf '# rustc=%s cargo=%s\n' "$(rustc --version)" "$(cargo --version)"

cargo run --release -p tenet --example operation_matrix --quiet \
    --no-default-features --features "$OP_MATRIX_CARGO_FEATURES"
