#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RAYON_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
: "${CARGO_TARGET_DIR:=target/operation-matrix}"
export RAYON_NUM_THREADS OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS CARGO_TARGET_DIR
export TENET_AUTHORITY="$(git describe --always --dirty)"
export TENFERRO_AUTHORITY="$(git -C ../tenferro-rs describe --always --dirty 2>/dev/null || printf unavailable)"

cargo run --release -p tenet --example operation_matrix --quiet
