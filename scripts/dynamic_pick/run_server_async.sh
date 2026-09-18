#!/usr/bin/env bash

source .venv/bin/activate

# The learner listens here; the rollout client (run_policy.sh) dials in directly, no SSH tunnel.
CLIENT_IP=0.0.0.0

# Device 0 samples, the rest update; batch_size must divide by the number of update devices.
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train_pi_robo_async.py \
    --config_task=configs/task/dynamic_pick.py \
    --dataset_path=./data/dynamic_pick/success \
    --num_data=25 \
    --batch_size=63 \
    --offline_ratio=0 \
    --config=configs/model/realtime_expo_ft_pi_config.py \
    --config.valids_keep_terminal_windows=True \
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
    --project_name=expo_ft_dynamic_pick \
    --output_dir=./checkpoints/dynamic_pick \
    --client_host="$CLIENT_IP" \
    --client_port=8103 \
    --fsdp_devices=1 \
    --delay=5 \
    --resume \
    --checkpoint_model \
    --checkpoint_buffer \
    --checkpoint_interval=10000 \
    --run_name=ours_dynamic_pick_async_25_delay5
