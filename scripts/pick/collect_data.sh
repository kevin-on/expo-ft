#!/usr/bin/env bash

source client/.venv/bin/activate

NUM_EPISODES=15

python -m client.collect_data \
    --save_root data/pick_cube_balance \
    --num_episodes $NUM_EPISODES \
    --task_config configs/task/pick.py
