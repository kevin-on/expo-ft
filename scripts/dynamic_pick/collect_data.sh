#!/usr/bin/env bash

source client/.venv/bin/activate

NUM_EPISODES=25  # stop after 25 successful episodes; 0 = run until stopped

python -m client.collect_data \
    --save_root data/dynamic_pick \
    --num_episodes $NUM_EPISODES \
    --task_config configs/task/dynamic_pick.py
