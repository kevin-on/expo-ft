#!/usr/bin/env bash
# From the appropriate login node: run_role.sh learner|inference RUN [PROFILE].
# Launches one step in an EXISTING allocation. Never allocates/cancels a parent job.
set -euo pipefail
ROLE=${1:?Usage: run_role.sh learner|inference RUN [PROFILE]}
shift
COMPUTE=false
if [[ "$ROLE" == --compute ]]; then COMPUTE=true; ROLE=${1:?}; shift; fi
case "$ROLE" in learner|inference) ;; *) echo 'Role must be learner or inference' >&2; exit 2;; esac
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_profile "$@"
SESSION=${SPLIT_SESSION:-$RUN}
[[ "$SESSION" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] || { echo "Invalid SPLIT_SESSION" >&2; exit 2; }
if [[ "$ROLE" == learner ]]; then
    JOB=$DELTA_JOB; NODE=$DELTA_NODE; STAGE=$DELTA_STAGE; IMAGE=$DELTA_IMAGE
    SOURCE=$DELTA_SOURCE; PERSIST=$DELTA_RUN
    CACHE_TAG=sync-EXPOLearner-n$DELTA_GPUS
    RESOURCES=(-c "$DELTA_CPUS" --mem="$DELTA_MEM" --gpus="$DELTA_GPUS")
else
    JOB=$ILIAD_JOB; NODE=$ILIAD_NODE; STAGE=$ILIAD_STAGE; IMAGE=$ILIAD_IMAGE
    SOURCE=$ILIAD_SOURCE; PERSIST=$ILIAD_RUN
    CACHE_TAG=split-inference-n1
    RESOURCES=(-c "$ILIAD_CPUS" --mem="$ILIAD_MEM" --gres=gpu:h200:1)
fi
if ! "$COMPUTE"; then
    [[ -z ${SLURM_JOB_ID:-} ]] || { echo 'Run this command on the login node, outside srun' >&2; exit 1; }
    squeue --steps -j "$JOB"
    scontrol show job "$JOB" -o
    echo "Starting $ROLE on $NODE in job $JOB; source=$SOURCE output=$PERSIST"
    exec srun --jobid="$JOB" --exact -N1 -n1 -w "$NODE" "${RESOURCES[@]}" \
        --time="$STEP_TIME" bash "$LAUNCH_DIR/run_role.sh" --compute "$ROLE" "$RUN" "$PROFILE"
fi
[[ ${SLURM_JOB_ID:-} == "$JOB" && $(hostname -s) == "$NODE" ]] || {
    echo "Expected Slurm job $JOB on $NODE" >&2; exit 1;
}
for file in "$IMAGE" "$SOURCE/train_pi_robo.py" "$PERSIST/link/transport-$ROLE.json" \
    "$PERSIST/link/token" "$PERSIST/link/cert.pem" "$PERSIST/link/key.pem" \
    "$STAGE/checkpoint/assets/$ASSET_ID/norm_stats.json"; do require_file "$file"; done
[[ -d "$STAGE/checkpoint/params" && -d "$STAGE/model-cache" ]] || exit 1
umask 077
# Hold one role per run; transport also refuses conflicting listener ports.
exec 9>"$PERSIST/.$ROLE.lock"
flock -n 9 || { echo "$ROLE is already running for $RUN" >&2; exit 1; }
if [[ "$ROLE" == learner ]]; then
    [[ -d "$DEMO" ]] || { echo "Missing HDF5 demos: $DEMO" >&2; exit 1; }
    require_file "$WANDB_KEY_FILE"
    if [[ -e "$PERSIST/checkpoints" ]]; then
        # A startup failure may leave only W&B's ID, with no trained state.
        # Archive that metadata; never overwrite checkpoints or replay records.
        if [[ -L "$PERSIST/checkpoints" || ! -d "$PERSIST/checkpoints" ]] || \
            [[ -n $(find "$PERSIST/checkpoints" -mindepth 1 ! -name wandb_id.txt -print -quit) ]]; then
            echo 'Saved training/replay exists; use an explicit resume or a fresh RUN' >&2; exit 1
        fi
        ARCHIVE=$(mktemp -d "$PERSIST/startup-retry.XXXXXX")
        mv "$PERSIST/checkpoints" "$ARCHIVE/checkpoints"
    fi
fi
umask 077
LOCAL=$(mktemp -d "/tmp/expo-split-$RUN.XXXXXX")
# Fresh IPC state on every launch. Both roles share SPLIT_SESSION from the profile.
printf "session=%s local=%s\n" "$SESSION" "$LOCAL"
mkdir -p "$LOCAL"/{home,cache,mailbox,runtime} "$STAGE/$JAX_CACHE_NAME" "$PERSIST/wandb"
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?Slurm must set CUDA_VISIBLE_DEVICES}
export APPTAINERENV_JAX_PLATFORMS=cuda APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export APPTAINERENV_XLA_PYTHON_CLIENT_MEM_FRACTION=.90 APPTAINERENV_EXPO_FSDP_DEVICES=$FSDP_DEVICES
export APPTAINERENV_OPENPI_DATA_HOME=/model-cache APPTAINERENV_XDG_CACHE_HOME=/cache
export APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_HF_DATASETS_OFFLINE=1
export APPTAINERENV_TOKENIZERS_PARALLELISM=false APPTAINERENV_OMP_NUM_THREADS=2
export APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_FASTVLA_GATHER_THREADS=12 APPTAINERENV_WANDB_MODE=disabled
export APPTAINERENV_PYTHONPATH=/opt/expo-ft:/opt/expo-ft/expo_ft/agents/vla/openpi/src:/opt/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client/src
BIND=$SOURCE:/opt/expo-ft:ro,$STAGE/checkpoint:/checkpoint:ro,$STAGE/model-cache:/model-cache
BIND+=,$PERSIST/link:/link:ro,$PERSIST:/output/$RUN,$LOCAL/cache:/cache,$LOCAL/mailbox:/mailbox
BIND+=,$STAGE/$JAX_CACHE_NAME:/home/kon/.cache/jax/$CACHE_TAG
[[ "$ROLE" != learner ]] || BIND+=,$DEMO:/demo:ro
GPU=(apptainer exec --nv --cleanenv --containall --no-mount cwd
    --home "$LOCAL/home:/home/kon" --workdir "$LOCAL/runtime" --pwd /opt/expo-ft
    --bind "$BIND" "$IMAGE")
for old_log in "$PERSIST/$ROLE.log" "$PERSIST/transport-$ROLE.log"; do
    if [[ -f "$old_log" ]]; then
        cp "$old_log" "$old_log.previous.$SLURM_JOB_ID.$SLURM_STEP_ID"
    fi
done
TRANSPORT_PID=
APP_PID=
cleanup() {
    trap - EXIT INT TERM
    [[ -z "$APP_PID" ]] || kill -TERM "$APP_PID" 2>/dev/null || true
    [[ -z "$APP_PID" ]] || wait "$APP_PID" 2>/dev/null || true
    [[ -z "$TRANSPORT_PID" ]] || kill -TERM "$TRANSPORT_PID" 2>/dev/null || true
    [[ -z "$TRANSPORT_PID" ]] || wait "$TRANSPORT_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${GPU[@]}" python -u -m expo_ft.distributed.transport --config "/link/transport-$ROLE.json" \
    >"$PERSIST/transport-$ROLE.log" 2>&1 &
TRANSPORT_PID=$!
for ((i=0; i<60; i++)); do
    kill -0 "$TRANSPORT_PID" 2>/dev/null || { echo "Transport exited; see $PERSIST/transport-$ROLE.log" >&2; exit 1; }
    [[ ! -f "$LOCAL/mailbox/transport-ready.json" ]] || break
    sleep 1
done
[[ -f "$LOCAL/mailbox/transport-ready.json" ]] || { echo 'Transport startup timeout' >&2; exit 1; }
ARGS=(--config=configs/model/expo_ft_pi_config.py --config_task=configs/task/pick.py
    --config.pi05_weight_loader_path=/checkpoint/params --config.pi05_assets_dir=/checkpoint/assets
    --config.pi05_asset_id="$ASSET_ID" --num_robot=2 --replan_steps=8 --delay=0 --update_type=episode
    --split_session="$SESSION" --split_mailbox=/mailbox --seed=42 --run_name="$RUN" --split_role="$ROLE")
if [[ "$ROLE" == learner ]]; then
    APPTAINERENV_WANDB_API_KEY=$(<"$WANDB_KEY_FILE")
    export APPTAINERENV_WANDB_API_KEY APPTAINERENV_WANDB_MODE=online
    export APPTAINERENV_WANDB_RUN_GROUP=$WANDB_GROUP APPTAINERENV_WANDB_DIR=/output/$RUN/wandb
    # train_pi_robo creates the checkpoint directory before initializing it.
    # The preflight above archives only startup metadata and refuses trained state.
    ARGS+=(--overwrite --dataset_path=/demo --batch_size="$BATCH_SIZE" --utd_ratio="$UTD_RATIO"
        --split_warmup_episodes="${WARMUP_EPISODES:-10}" --num_updates="$NUM_UPDATES" --fsdp_devices="$FSDP_DEVICES" --offline_ratio=0
        --max_steps="$MAX_STEPS" --checkpoint_model --checkpoint_buffer
        --checkpoint_interval="$CHECKPOINT_INTERVAL" --project_name="$WANDB_PROJECT" --output_dir=/output)
else
    ARGS+=(--client_host=0.0.0.0 --client_port="$CLIENT_PORT" --output_dir="$WS_VIDEO_ROOT")
fi
# Keep the application PID (not a tee pipeline PID) for signal cleanup.
"${GPU[@]}" python -u train_pi_robo.py "${ARGS[@]}" > >(tee "$PERSIST/$ROLE.log") 2>&1 &
APP_PID=$!
set +e
wait "$APP_PID"
status=$?
set -e
APP_PID=
exit "$status"
