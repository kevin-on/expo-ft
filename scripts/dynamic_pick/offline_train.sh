#!/usr/bin/env bash

source .venv/bin/activate

python train_offline_rtc.py \
    --config_task=configs/task/dynamic_pick.py \
    --dataset_path=./data/dynamic_pick/success \
    --num_data=25 \
    --batch_size=64 \
    --fsdp_devices=1 \
    --config=configs/model/rtc_pi_config.py \
    --config.freeze_pi05_encoder=False \
    --config.p1_use_prefix_conditioning=True \
    --config.p1_max_delay=10 \
    --config.pi05_config_name=expo_pi05_droid_lora_finetune_sft_cartesian_state \
    --config.pi05_assets_dir="./assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
    --config.pi05_asset_id="expo_ft/droid_dynamic_pick_25" \
    --project_name=pi_sft_dynamic_pick \
    --output_dir=./checkpoints/dynamic_pick_rtc_offline \
    --max_steps=4000 \
    --resume \
    --checkpoint_model \
    --checkpoint_interval=2000 \
    --run_name=pi_rtc_dynamic_pick_25_non_frozen_maxdelay10
