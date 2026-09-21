#!/bin/bash
# Invoked inside the container by run.sh validate.
set -euo pipefail
[[ $# == 0 ]] || { echo 'Configure validation with EXPO_* environment variables, not arguments.' >&2; exit 2; }
read -r -a batches <<< "${EXPO_BATCH_SIZES:-8 64}"
[[ ${#batches[@]} -gt 0 ]] || exit 2
for value in "${batches[@]}" "${EXPO_UPDATES:-20}" "${EXPO_UTD:-20}" "${EXPO_FSDP_DEVICES:-1}"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Expected positive integer, got: $value" >&2; exit 2; }
done
num_devices=$(python -c 'import jax; print(len(jax.devices()))')
finish() {
    rc=$?
    trap - EXIT
    # Preserve the training/restore failure even when summarizing an incomplete run.
    python /tools/summarize.py /output > /output/summary.stdout.txt || { if [[ $rc == 0 ]]; then rc=1; fi; }
    exit "$rc"
}
trap finish EXIT
for batch in "${batches[@]}"; do
    label=batch-$batch
    [[ ! -e /output/$label ]] || { echo "Repeated batch or existing output: $label" >&2; exit 2; }
    for phase in train restore; do
        printf 'Starting %s-%s at %s\n' "$label" "$phase" "$(date -Is)" | tee -a /output/run-status.txt
        set +e
        timeout --signal=TERM --kill-after=60s "${EXPO_PHASE_TIMEOUT:-100m}" python tests/gpu/learner_smoke.py \
          --dataset /demo --params /openpi-cache/openpi-assets/checkpoints/pi05_base/params \
          --output /output/$label --num-devices "$num_devices" --fsdp-devices "${EXPO_FSDP_DEVICES:-1}" \
          --batch-size "$batch" --utd-ratio "${EXPO_UTD:-20}" --updates "${EXPO_UPDATES:-20}" --phase "$phase" \
          > /output/$label-$phase.log 2>&1
        rc=$?
        set -e
        printf '%s-%s exit=%s at %s\n' "$label" "$phase" "$rc" "$(date -Is)" | tee -a /output/run-status.txt
        if [[ $rc != 0 ]]; then exit "$rc"; fi
    done
done
