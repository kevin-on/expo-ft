#!/bin/bash
set -euo pipefail
[[ "${SLURM_JOB_ID:?Use srun inside the held allocation}" == 17539504 ]]
STAGE=/tmp/kevinon/expo-real-17539504
SNAPSHOT=/iris/u/kevinon/workspace/expo-ft-real/17539504
RESULTS=/iris/u/kevinon/outputs/expo-ft-real/17539504-robot1
source "$STAGE/container.sh"
export APPTAINERENV_WANDB_RUN_ID=66d7e522
export APPTAINERENV_WANDB_RUN_GROUP=pick-pi05base-r1-20260921
export APPTAINERENV_WANDB_TAGS=pi05_base,robot1-only,a40x2,batch8,utd20,seed42
mkdir -p "$RESULTS"
exec 9>"$STAGE/learner.lock"
flock -n 9 || { echo 'A learner already owns this allocation'; exit 1; }
child_pid=""
backup_pid=""
cleanup() {
 rc=$?
 trap - EXIT TERM INT
 if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
   kill -TERM "$child_pid" 2>/dev/null || true
   wait "$child_pid" 2>/dev/null || true
 fi
 if [[ -n "$backup_pid" ]]; then kill "$backup_pid" 2>/dev/null || true; wait "$backup_pid" 2>/dev/null || true; fi
 python3 "$SNAPSHOT/snapshot_robot1.py" || true
 cp "$STAGE/output/learner-robot1.log" "$RESULTS/learner-robot1.log" || true
 cp -R "$STAGE/output/wandb" "$RESULTS/" 2>/dev/null || true
 printf '%s\n' "$rc" > "$RESULTS/learner-exit-code.txt"
 exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Keep hot logs/replay/checkpoints on node-local storage. Periodic recovery is a
# single archive on shared storage, containing only atomically committed files.
(while sleep 600; do python3 "$SNAPSHOT/snapshot_robot1.py"; done) > "$STAGE/output/backup-robot1.log" 2>&1 &
backup_pid=$!
printf 'Starting learner in job %s on %s; logs %s\n' "$SLURM_JOB_ID" "$(hostname)" "$STAGE/output/learner-robot1.log"
"${container[@]}" python -u train_pi_robo.py \
 --config_task=configs/task/pick.py --config=configs/model/expo_ft_pi_config.py \
 --dataset_path=/demo --num_data=1 --offline_ratio=0 \
 --config.pi05_weight_loader_path=/openpi-cache/openpi-assets/checkpoints/pi05_base/params \
 --config.pi05_assets_dir=/assets --config.pi05_asset_id=validation-recording \
 --num_robot=1 --client_host=0.0.0.0 --client_port=8103 \
 --batch_size=8 --utd_ratio=20 --fsdp_devices=2 \
 --update_type=episode --num_updates=3 --delay=0 --replan_steps=8 \
 --max_steps=10000 --notqdm \
 --output_dir=/scr/kevinon/data/expo-real-17539504 --run_name=pick-robot1-a40x2-b8-utd20-s42-j17539504 \
 --project_name=mtexpo --resume --checkpoint_model --checkpoint_buffer --checkpoint_interval=1000 \
 > "$STAGE/output/learner-robot1.log" 2>&1 &
child_pid=$!
set +e
wait "$child_pid"
code=$?
set -e
child_pid=""
exit "$code"
