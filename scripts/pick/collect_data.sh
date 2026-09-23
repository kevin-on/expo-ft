#!/usr/bin/env bash
set -euo pipefail

source client/.venv/bin/activate

ROBOT_ID=${ROBOT_ID:-0}
case "$ROBOT_ID" in
    0|1) ;;
    *) echo "ROBOT_ID must be 0 or 1" >&2; exit 2 ;;
esac
NUM_EPISODES=${NUM_EPISODES:-15}
SAVE_ROOT=${SAVE_ROOT:-data/pick_cube_balance/robot${ROBOT_ID}}

exec python -m client.collect_data \
    --save_root "$SAVE_ROOT" \
    --num_episodes "$NUM_EPISODES" \
    --task_config configs/task/pick.py \
    --robot_config "configs/robots/robot-${ROBOT_ID}.json" \
    "$@"
