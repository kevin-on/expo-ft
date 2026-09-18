#!/usr/bin/env bash

source .venv/bin/activate

MAX_EPISODES=25
TASK_CONFIG="configs/task/dynamic_pick.py"
DATA_DIR="./data/dynamic_pick/success"
REPO_NAME="expo_ft/droid_dynamic_pick_${MAX_EPISODES}"

uv run scripts/convert_droid_data_to_lerobot.py \
    --data_dir="$DATA_DIR" \
    --repo_name="$REPO_NAME" \
    --task_config="$TASK_CONFIG" \
    --max_episodes="$MAX_EPISODES" \
    --use_cartesian_state \
    --no-push-to-hub
