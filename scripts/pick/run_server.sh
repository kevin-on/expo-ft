#!/usr/bin/env bash

source .venv/bin/activate

# The learner listens here; the rollout client (run_policy.sh) dials in directly, no SSH tunnel.
CLIENT_IP=0.0.0.0

export CUDA_VISIBLE_DEVICES=0,1,2,3

python train_pi_robo.py \
    --config_task=configs/task/pick.py \
    --dataset_path=./data/pick_cube_balance/success \
    --num_data=10 \
    --update_type=episode \
    --num_updates=3 \
    --offline_ratio=0 \
    --config=configs/model/expo_ft_pi_config.py \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_scale=0.2 \
    --initial_sft_checkpoint="${SFT_CHECKPOINT:?Set a completed SFT checkpoint}" \
    --initial_sft_base="${SFT_BASE_PARAMS:-}" \
    --project_name=expo_ft_pick \
    --output_dir=./checkpoints/pick \
    --client_host="$CLIENT_IP" \
    --client_port=8102 \
    --fsdp_devices=1 \
    --checkpoint_model \
    --checkpoint_buffer \
    --checkpoint_interval=2000 \
    --run_name=expo_pick_example
