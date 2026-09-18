#!/usr/bin/env bash

source .venv/bin/activate

MAX_EPISODES=10
TASK_CONFIG="configs/task/pick.py"
DATA_DIR="./data/pick_cube_balance/success"
REPO_NAME="expo_ft/droid_pick_cube_${MAX_EPISODES}"

uv run scripts/convert_droid_data_to_lerobot.py \
    --data_dir="$DATA_DIR" \
    --repo_name="$REPO_NAME" \
    --task_config="$TASK_CONFIG" \
    --max_episodes="$MAX_EPISODES" \
    --use_cartesian_state \
    --no-push-to-hub
