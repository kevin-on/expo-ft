#!/usr/bin/env bash

source .venv/bin/activate
CLIENT_IP=0.0.0.0

export CUDA_VISIBLE_DEVICES=0

python eval_droid_policy.py \
    --config_task=configs/task/pick.py \
    --client_host="$CLIENT_IP" \
    --client_port=8102 \
    --checkpoint_kind=online --checkpoint_dir="${ONLINE_CHECKPOINT:?Set a completed online checkpoint step}" \
    --num_episodes=35
