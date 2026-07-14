#!/usr/bin/env bash
set -euo pipefail

label="$1"
shift
samples="${RUNNER_TEMP:-/tmp}/tenet-disk-headroom-${label}.tsv"

sample_disk() {
  df -Pk / | awk -v now="$(date +%s)" 'NR == 2 { print now "\t" $2 "\t" $4 }' >> "$samples"
}

: > "$samples"
sample_disk

"$@" &
command_pid=$!

(
  while kill -0 "$command_pid" 2>/dev/null; do
    sample_disk
    sleep 2
  done
) &
sampler_pid=$!

set +e
wait "$command_pid"
status=$?
set -e
wait "$sampler_pid" || true
sample_disk

summary="$({
  awk -v label="$label" '
    NR == 1 { total = $2; initial = $3 }
    min == "" || $3 < min { min = $3 }
    { post = $3; count += 1 }
    END {
      printf "DISK_HEADROOM label=%s root_gib=%.2f initial_available_gib=%.2f min_available_gib=%.2f post_available_gib=%.2f samples=%d", label, total / 1048576, initial / 1048576, min / 1048576, post / 1048576, count
    }
  ' "$samples"
})"

printf '%s\n' "$summary"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  printf '%s\n' "$summary" >> "$GITHUB_STEP_SUMMARY"
fi

exit "$status"
