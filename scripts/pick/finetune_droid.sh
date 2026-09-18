#!/usr/bin/env bash

source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_ID="droid_pick_cube_10"
REPO_ID="expo_ft/${DATA_ID}"
ASSETS_DIR="./assets/expo_pi05_droid_lora_finetune_sft_cartesian_state"
ASSET_ID="expo_ft/droid_pick_cube_10"

uv run expo_ft/agents/vla/openpi/scripts/train.py expo_pi05_droid_lora_finetune_sft_cartesian_state \
    --exp-name="${DATA_ID}_lora_sft" \
    --resume \
    --data.repo_id="$REPO_ID" \
    --data.assets.assets_dir="$ASSETS_DIR" \
    --data.assets.asset_id="$ASSET_ID" \
    --num_train_steps=4001 \
    --save_interval=2000 \
    --fsdp_devices=1
