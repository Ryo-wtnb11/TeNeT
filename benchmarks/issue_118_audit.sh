#!/bin/sh
set -eu

: "${FINITE_TORUS_DIR:?set FINITE_TORUS_DIR to finite-torus/migrate}"
: "${TENFERRO_DIR:?set TENFERRO_DIR to the tenferro checkout used by TeNeT}"
RUNS="${RUNS:-5}"
WARMLOOP="${WARMLOOP:-40}"
CHIS="${CHIS:-16 32}"
tenet_root=$(git rev-parse --show-toplevel)

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        printf 'sha256sum or shasum is required\n' >&2
        return 1
    fi
}

metadata_file=$(mktemp)
time_file=$(mktemp)
trap 'rm -f "$metadata_file" "$time_file"' EXIT HUP INT TERM
cargo metadata --format-version 1 >"$metadata_file"

if ! tenferro_manifest=$(python3 -c '
import json, sys
data = json.load(open(sys.argv[1]))
paths = sorted(p["manifest_path"] for p in data["packages"] if p["name"].startswith("tenferro-"))
if not paths:
    raise SystemExit("Cargo metadata resolved no tenferro packages")
print(paths[0])
' "$metadata_file"); then
    printf 'failed to resolve tenferro provenance from Cargo metadata\n' >&2
    exit 1
fi
if ! resolved_tenferro_root=$(git -C "$(dirname "$tenferro_manifest")" rev-parse --show-toplevel); then
    printf 'resolved tenferro package is not in a git checkout: %s\n' "$tenferro_manifest" >&2
    exit 1
fi
if ! requested_tenferro_root=$(git -C "$TENFERRO_DIR" rev-parse --show-toplevel); then
    printf 'TENFERRO_DIR is not a valid git checkout: %s\n' "$TENFERRO_DIR" >&2
    exit 1
fi
if [ "$(cd "$resolved_tenferro_root" && pwd -P)" != "$(cd "$requested_tenferro_root" && pwd -P)" ]; then
    printf 'resolved tenferro %s does not match TENFERRO_DIR %s\n' \
        "$resolved_tenferro_root" "$requested_tenferro_root" >&2
    exit 1
fi
if ! tenferro_sha=$(git -C "$resolved_tenferro_root" rev-parse HEAD); then
    printf 'failed to read resolved tenferro SHA\n' >&2
    exit 1
fi

printf 'tenet_sha=%s\n' "$(git rev-parse HEAD)"
printf 'tenferro_root=%s tenferro_sha=%s\n' "$resolved_tenferro_root" "$tenferro_sha"
printf 'tenet_cargo_lock_sha256=%s\n' "$(sha256_file "$tenet_root/Cargo.lock")"
printf 'rustc=%s\n' "$(rustc -Vv | tr '\n' ' ')"
printf 'system=%s\n' "$(uname -a)"
printf 'runs=%s warmloop=%s chis=%s\n' "$RUNS" "$WARMLOOP" "$CHIS"
printf 'RAYON_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s\n' \
    "${RAYON_NUM_THREADS:-unset}" "${OPENBLAS_NUM_THREADS:-unset}" \
    "${OMP_NUM_THREADS:-unset}" "${MKL_NUM_THREADS:-unset}"

cargo test --release -p tenet-core hom_space_id_remains_semantic_after_intern_eviction \
    --lib --no-run --message-format=json >"$metadata_file"
test_binary=$(python3 -c '
import json, sys
paths = []
for line in open(sys.argv[1]):
    record = json.loads(line)
    if record.get("reason") == "compiler-artifact" and record["target"]["name"] == "tenet_core":
        executable = record.get("executable")
        if executable:
            paths.append(executable)
if not paths:
    raise SystemExit("tenet-core release test executable missing from Cargo output")
print(paths[-1])
' "$metadata_file")
if [ -z "$test_binary" ]; then
    printf 'tenet-core release test binary not found\n' >&2
    exit 1
fi
sample=1
while [ "$sample" -le "$RUNS" ]; do
    case "$(uname -s)" in
        Darwin)
            /usr/bin/time -l "$test_binary" no_such_test_filter 2>"$time_file" >/dev/null
            baseline_rss=$(awk '/maximum resident set size/ {print $1}' "$time_file")
            /usr/bin/time -l "$test_binary" hom_space_id_remains_semantic_after_intern_eviction \
                2>"$time_file" >/dev/null
            churn_rss=$(awk '/maximum resident set size/ {print $1}' "$time_file")
            unit=bytes
            ;;
        Linux)
            /usr/bin/time -f '%M' -o "$time_file" "$test_binary" no_such_test_filter >/dev/null
            baseline_rss=$(cat "$time_file")
            /usr/bin/time -f '%M' -o "$time_file" "$test_binary" \
                hom_space_id_remains_semantic_after_intern_eviction >/dev/null
            churn_rss=$(cat "$time_file")
            unit=KiB
            ;;
        *) printf 'RSS sampling supports Darwin and Linux only\n' >&2; exit 1 ;;
    esac
    printf 'metadata_rss sample=%s baseline=%s churn_9000=%s unit=%s\n' \
        "$sample" "$baseline_rss" "$churn_rss" "$unit"
    sample=$((sample + 1))
done

if ! finite_sha=$(git -C "$FINITE_TORUS_DIR" rev-parse HEAD); then
    printf 'FINITE_TORUS_DIR is not inside a git checkout: %s\n' "$FINITE_TORUS_DIR" >&2
    exit 1
fi
cd "$FINITE_TORUS_DIR"
cargo metadata --format-version 1 >"$metadata_file"
case "$(cat "$metadata_file")" in
    *"$tenet_root/tenet/Cargo.toml"*) ;;
    *) printf 'finite-torus does not resolve tenet from %s\n' "$tenet_root" >&2; exit 1 ;;
esac
finite_workspace=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["workspace_root"])' "$metadata_file")
finite_lock="$finite_workspace/Cargo.lock"
if [ ! -f "$finite_lock" ]; then
    printf 'finite-torus governing Cargo.lock not found: %s\n' "$finite_lock" >&2
    exit 1
fi
printf 'finite_torus_sha=%s finite_torus_lock_sha256=%s\n' \
    "$finite_sha" "$(sha256_file "$finite_lock")"
cargo tree -e features -p tenet >"$metadata_file"
printf 'feature_tree_sha256=%s\n' "$(sha256_file "$metadata_file")"
cargo build --release --example bench_energy
for chi in $CHIS; do
    run=1
    while [ "$run" -le "$RUNS" ]; do
        printf 'run=%s chi=%s\n' "$run" "$chi"
        CHI="$chi" WARMLOOP="$WARMLOOP" target/release/examples/bench_energy 2>&1
        run=$((run + 1))
    done
done
