#!/usr/bin/env bash
# Run inside the allocated GPU/container environment. Does not submit a job.
set -euo pipefail
robot_index=${1:?Usage: eval_sft_policy.sh 0|1}
case "$robot_index" in 0|1) ;; *) echo 'Robot index must be 0 or 1' >&2; exit 2;; esac
: "${SFT_CHECKPOINT:?Set the completed OpenPI step directory on the GPU host}"
: "${EXPO_EVAL_OUTPUT_DIR:?Set a new result directory on the GPU host}"
: "${EXPO_CLIENT_VIDEO_DIR:?Set an absolute video directory on the workstation}"
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
# A fresh directory avoids overwriting results from a previous checkpoint/run.
mkdir "$EXPO_EVAL_OUTPUT_DIR"
"${EXPO_EVAL_PYTHON:-python}" -u eval_droid_policy.py \
    --config_task=configs/task/pick.py \
    --checkpoint_kind=sft --checkpoint_dir="$SFT_CHECKPOINT" \
    --mirror_y="$([[ "$robot_index" == 1 ]] && echo true || echo false)" \
    --only_base_actions \
    --client_host=0.0.0.0 --client_port="$(( ${EXPO_EVAL_BASE_PORT:-8202} + robot_index ))" \
    --client_video_dir="$EXPO_CLIENT_VIDEO_DIR" \
    --num_episodes="${EXPO_EVAL_EPISODES:-10}" \
    --replan_steps="${EXPO_REPLAN_STEPS:-8}" --delay=0 \
    --seed="${EXPO_EVAL_SEED:-42}" 2>&1 | tee "$EXPO_EVAL_OUTPUT_DIR/eval.log"
