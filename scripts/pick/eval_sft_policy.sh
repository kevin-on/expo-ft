#!/usr/bin/env bash
# Run inside the allocated GPU/container environment. Does not submit a job.
set -euo pipefail
robot_index=${1:?Usage: eval_sft_policy.sh 0|1}
case "$robot_index" in 0|1) ;; *) echo 'Robot index must be 0 or 1' >&2; exit 2;; esac
: "${SFT_CHECKPOINT:?Set the completed OpenPI step directory on the GPU host}"
: "${SFT_ASSET_ID:?Set the checkpoint stats ID, e.g. expo_ft/pick_mixed_100}"
: "${EXPO_DATASET:?Set an HDF5 episode-parent directory on the GPU host (shape initialization only)}"
: "${EXPO_EVAL_OUTPUT_DIR:?Set a new result directory on the GPU host}"
: "${EXPO_CLIENT_VIDEO_DIR:?Set an absolute video directory on the workstation}"
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
# A fresh directory avoids overwriting results from a previous checkpoint/run.
mkdir "$EXPO_EVAL_OUTPUT_DIR"
"${EXPO_EVAL_PYTHON:-python}" -u eval_droid_policy.py \
    --config_task=configs/task/pick.py \
    --config=configs/model/expo_ft_pi_config.py \
    --config.pi05_weight_loader_path="$SFT_CHECKPOINT/params" \
    --config.pi05_assets_dir="$SFT_CHECKPOINT/assets" \
    --config.pi05_asset_id="$SFT_ASSET_ID" \
    --config.N=1 --config.n_edit_samples=0 \
    --mirror_y="$([[ "$robot_index" == 1 ]] && echo true || echo false)" \
    --dataset_path="$EXPO_DATASET" --num_data=1 \
    --only_base_actions --checkpoint_step=0 \
    --client_host=0.0.0.0 --client_port="$(( ${EXPO_EVAL_BASE_PORT:-8202} + robot_index ))" \
    --client_video_dir="$EXPO_CLIENT_VIDEO_DIR" \
    --num_episodes="${EXPO_EVAL_EPISODES:-10}" \
    --replan_steps="${EXPO_REPLAN_STEPS:-8}" --delay=0 --fsdp_devices=1 \
    --seed="${EXPO_EVAL_SEED:-42}" 2>&1 | tee "$EXPO_EVAL_OUTPUT_DIR/eval.log"
