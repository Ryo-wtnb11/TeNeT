#!/bin/sh
set -eu

: "${FINITE_TORUS_DIR:?set FINITE_TORUS_DIR to finite-torus/migrate}"
: "${TENFERRO_DIR:?set TENFERRO_DIR to the tenferro checkout used by TeNeT}"
RUNS="${RUNS:-5}"
WARMLOOP="${WARMLOOP:-40}"
CHIS="${CHIS:-16 32}"
tenet_root=$(git rev-parse --show-toplevel)

printf 'tenet_sha=%s\n' "$(git rev-parse HEAD)"
printf 'tenferro_sha=%s\n' "$(git -C "$TENFERRO_DIR" rev-parse HEAD)"
printf 'cargo_lock_sha256=%s\n' "$(shasum -a 256 Cargo.lock | awk '{print $1}')"
printf 'rustc=%s\n' "$(rustc -Vv | tr '\n' ' ')"
printf 'system=%s\n' "$(uname -a)"
printf 'runs=%s warmloop=%s chis=%s\n' "$RUNS" "$WARMLOOP" "$CHIS"
printf 'RAYON_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s\n' \
    "${RAYON_NUM_THREADS:-unset}" "${OPENBLAS_NUM_THREADS:-unset}" \
    "${OMP_NUM_THREADS:-unset}" "${MKL_NUM_THREADS:-unset}"

cargo build --release -p tenet-core --example homspace_identity_churn
for count in 0 8192 9000; do
    target/release/examples/homspace_identity_churn "$count" 5 &
    pid=$!
    sleep 1
    rss_kib=$(ps -o rss= -p "$pid" | tr -d ' ')
    printf 'homspace_metadata count=%s rss_kib=%s\n' "$count" "$rss_kib"
    wait "$pid"
done

cd "$FINITE_TORUS_DIR"
case "$(cargo metadata --no-deps --format-version 1)" in
    *"$tenet_root/tenet/Cargo.toml"*) ;;
    *) printf 'finite-torus does not resolve tenet from %s\n' "$tenet_root" >&2; exit 1 ;;
esac
printf 'feature_tree_sha256=%s\n' \
    "$(cargo tree -e features -p tenet | shasum -a 256 | awk '{print $1}')"
cargo build --release --example bench_energy
for chi in $CHIS; do
    run=1
    while [ "$run" -le "$RUNS" ]; do
        printf 'run=%s chi=%s\n' "$run" "$chi"
        CHI="$chi" WARMLOOP="$WARMLOOP" \
            target/release/examples/bench_energy 2>&1
        run=$((run + 1))
    done
done
