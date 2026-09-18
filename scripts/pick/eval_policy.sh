#!/usr/bin/env bash

source .venv/bin/activate
CLIENT_IP=0.0.0.0

export CUDA_VISIBLE_DEVICES=0,1,2,3

python eval_droid_policy.py \
    --config_task=configs/task/pick.py \
    --config=configs/model/expo_ft_pi_config.py \
    --dataset_path=./data/pick_cube_balance/success \
    --num_data=1 \
    --client_host="$CLIENT_IP" \
    --client_port=8102 \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_scale=0.2 \
    --config.pi05_config_name=expo_pi05_droid_lora_finetune_sft_cartesian_state \
    --config.pi05_weight_loader_path="./checkpoints/expo_pi05_droid_lora_finetune_sft_cartesian_state/droid_pick_cube_10_lora_sft/2000/params" \
    --config.pi05_assets_dir="./assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
    --config.pi05_asset_id="expo_ft/droid_pick_cube_10" \
    --checkpoint_dir=./checkpoints/pick/expo_pick_example/checkpoints \
    --checkpoint_step=4000 \
    --num_episodes=35
