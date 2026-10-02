#!/usr/bin/env bash

source .venv/bin/activate

# The learner listens here; the rollout client (run_policy.sh) dials in directly, no SSH tunnel.
CLIENT_IP=0.0.0.0

export CUDA_VISIBLE_DEVICES=0,1,2

python train_pi_robo_async.py \
    --config_task=configs/task/pick.py \
    --dataset_path=./data/pick_cube_balance/success \
    --num_data=10 \
    --offline_ratio=0 \
    --config=configs/model/expo_ft_pi_config.py \
    --initial_sft_checkpoint="${SFT_CHECKPOINT:?Set a completed SFT checkpoint}" \
    --initial_sft_base="${SFT_BASE_PARAMS:-}" \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_scale=0.2 \
    --project_name=expo_ft_pick \
    --output_dir=./checkpoints/pick \
    --client_host="$CLIENT_IP" \
    --client_port=8102 \
    --fsdp_devices=1 \
    --checkpoint_model \
    --checkpoint_buffer \
    --checkpoint_interval=4000 \
    --run_name=expo_pick_example
