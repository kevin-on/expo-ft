#!/usr/bin/env bash

source .venv/bin/activate
CLIENT_IP=0.0.0.0

export CUDA_VISIBLE_DEVICES=0,1,2,3

python eval_droid_policy.py \
    --config_task=configs/task/dynamic_pick.py \
    --config=configs/model/realtime_expo_ft_pi_config.py \
    --dataset_path=./data/dynamic_pick/success \
    --num_data=1 \
    --client_host="$CLIENT_IP" \
    --client_port=8103 \
    --config.N=32 \
    --config.n_edit_samples=32 \
    --config.filter_N=32 \
    --config.filter_n_edit=1 \
    --config.edit_scale=0.1 \
    --config.filter_add_delayed_obs=True \
    --config.pi05_config_name=expo_pi05_droid_lora_finetune_sft_cartesian_state \
    --config.pi05_weight_loader_path="./checkpoints/dynamic_pick_rtc_offline/pi_rtc_dynamic_pick_25_non_frozen_maxdelay10/checkpoints/2000/params" \
    --config.pi05_assets_dir="./assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
    --config.pi05_asset_id="expo_ft/droid_dynamic_pick_25" \
    --checkpoint_dir=./checkpoints/dynamic_pick/ours_dynamic_pick_25_delay5/checkpoints \
    --delay=5 \
    --num_episodes=60
