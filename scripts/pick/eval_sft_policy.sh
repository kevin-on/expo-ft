#!/usr/bin/env bash
# One shared GPU policy and a Space-gated terminal dashboard.
set -euo pipefail
selection=${1:-both}
case "$selection" in both) robots=(0 1);; 0|1) robots=("$selection");; *) echo 'Usage: eval_sft_policy.sh [both|0|1]' >&2; exit 2;; esac
kind=${EXPO_EVAL_KIND:-sft}
checkpoint=${EXPO_EVAL_CHECKPOINT:-${SFT_CHECKPOINT:-}}
: "${checkpoint:?Set EXPO_EVAL_CHECKPOINT or SFT_CHECKPOINT to a completed step}"
: "${EXPO_EVAL_OUTPUT_DIR:?Set a new output directory on the GPU host}"
: "${EXPO_CLIENT_VIDEO_DIR:?Set an absolute video root on the workstation}"
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
args=(--checkpoint-dir "$checkpoint" --policy-kind "$kind" --robots "${robots[@]}"
      --output-dir "$EXPO_EVAL_OUTPUT_DIR" --client-video-dir "$EXPO_CLIENT_VIDEO_DIR"
      --host "${EXPO_EVAL_BIND_HOST:-0.0.0.0}" --base-port "${EXPO_EVAL_BASE_PORT:-8202}"
      --episodes "${EXPO_EVAL_EPISODES:-10}" --seed "${EXPO_EVAL_SEED:-42}")
[[ -z ${EXPO_REPLAN_STEPS:-} ]] || args+=(--replan-steps "$EXPO_REPLAN_STEPS")
[[ -z ${INITIAL_SFT_CHECKPOINT:-} ]] || args+=(--initial-sft-checkpoint "$INITIAL_SFT_CHECKPOINT")
exec "${EXPO_EVAL_PYTHON:-python}" -u eval_sft_robots.py "${args[@]}"
