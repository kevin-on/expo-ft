#!/bin/bash
# Run directly in an allocated Delta compute shell, or via the sbatch wrapper.
set -euo pipefail
if [[ "${1:-}" == --help || $# == 0 ]]; then
    cat <<'HELP'
Usage: bash docker/learner/delta/run.sh validate|preflight|COMMAND [ARGS...]
Required: EXPO_INPUT_ROOT (SIF, openpi-cache, demo), EXPO_OUTPUT_DIR (new directory).
Optional: EXPO_IMAGE, EXPO_IMAGE_SHA256, EXPO_DATASET, EXPO_BATCH_SIZES="8 64",
EXPO_UPDATES=20, EXPO_UTD=20, EXPO_FSDP_DEVICES=1, EXPO_PHASE_TIMEOUT=100m,
EXPO_MEMORY_FRACTION=0.95, EXPO_WANDB_MODE=disabled.
Run inside a Slurm allocation. GPU/CPU requests belong to srun or sbatch, not here.
No model/library installation, robot commands, or additional allocations are made.
HELP
    exit 0
fi
: "${SLURM_JOB_ID:?Run on an allocated compute node}"
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU visibility is required}"
: "${EXPO_INPUT_ROOT:?Set the persistent input directory}"
: "${EXPO_OUTPUT_DIR:?Set a new persistent results directory}"
[[ $(uname -m) == x86_64 ]] || { echo 'This image targets Delta x86-64, not DeltaAI ARM.' >&2; exit 2; }
for tool in apptainer python3 git nvidia-smi sha256sum; do command -v "$tool" >/dev/null; done
TOOLS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$TOOLS" rev-parse --show-toplevel)
INPUT=$(realpath -e "$EXPO_INPUT_ROOT")
IMAGE=$(realpath -e "${EXPO_IMAGE:-$INPUT/expo-ft-learner-jax053.sif}")
DATASET=$(realpath -e "${EXPO_DATASET:-$INPUT/demo}")
[[ -f "$INPUT/openpi-cache/manifest.json" ]] || { echo 'Missing input checksum manifest' >&2; exit 2; }
[[ -d "$DATASET" ]] || { echo 'Dataset must be the episode-parent directory' >&2; exit 2; }
# Atomically refuse existing results, including previous failed runs.
mkdir -p -- "$(dirname -- "$EXPO_OUTPUT_DIR")"
mkdir -- "$EXPO_OUTPUT_DIR"
OUTPUT=$(realpath -e "$EXPO_OUTPUT_DIR")
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/expo-delta-${SLURM_JOB_ID}-XXXXXX")
printf '%s\n' "$STAGE" > "$OUTPUT/local-stage.txt"
monitor_pid=''
child_pid=''
finish() {
    rc=$?
    trap - EXIT TERM INT
    if [[ -n "$child_pid" ]]; then kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true; fi
    if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
    printf '%s\n' "$rc" > "$OUTPUT/exit-code.txt"
    printf 'Finished exit=%s; results=%s; local scratch=%s\n' "$rc" "$OUTPUT" "$STAGE"
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
exec > >(tee "$OUTPUT/launcher.log") 2>&1
mkdir -p "$STAGE"/{home,runtime,cache,apptainer-cache,apptainer-tmp,jax-cache,source,tools} "$OUTPUT/jax-cache"
export APPTAINER_CACHEDIR="$STAGE/apptainer-cache" APPTAINER_TMPDIR="$STAGE/apptainer-tmp"
{ date -Is; hostname; apptainer --version; nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv; df -h "$STAGE"; } > "$OUTPUT/host.txt"
printf 'job=%s\ncpus=%s\ngpus=%s\n' "$SLURM_JOB_ID" "${SLURM_CPUS_PER_TASK:-1}" "$CUDA_VISIBLE_DEVICES" > "$OUTPUT/allocation.txt"
# Store only explicitly selected runtime settings, never the process environment/credentials.
printf 'batch_sizes=%s\nupdates=%s\nutd=%s\nfsdp=%s\nmemory_fraction=%s\n' "${EXPO_BATCH_SIZES:-8 64}" "${EXPO_UPDATES:-20}" "${EXPO_UTD:-20}" "${EXPO_FSDP_DEVICES:-1}" "${EXPO_MEMORY_FRACTION:-0.95}" > "$OUTPUT/settings.txt"
cp -- "$IMAGE" "$STAGE/learner.sif"
sha256sum "$STAGE/learner.sif" > "$OUTPUT/image-sha256.txt"
if [[ -n "${EXPO_IMAGE_SHA256:-}" ]]; then
    printf '%s  %s\n' "$EXPO_IMAGE_SHA256" "$STAGE/learner.sif" | sha256sum -c -
fi
python3 "$TOOLS/snapshot.py" "$REPO" "$STAGE/source" "$OUTPUT/source-manifest.json"
cp "$TOOLS"/*.py "$TOOLS"/*.sh "$STAGE/tools/"
cp "$REPO/docker/learner/verify_assets.py" "$STAGE/tools/"
cp -R "$STAGE/tools" "$OUTPUT/tools"
cp -R "$STAGE/source" "$OUTPUT/source"
cp -R "$INPUT/openpi-cache" "$STAGE/openpi-cache"
cp -R "$DATASET" "$STAGE/demo"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export APPTAINERENV_JAX_PLATFORMS=cuda APPTAINERENV_JAX_LOG_COMPILES=1 APPTAINERENV_JAX_EXPLAIN_CACHE_MISSES=1
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE=false APPTAINERENV_XLA_PYTHON_CLIENT_MEM_FRACTION="${EXPO_MEMORY_FRACTION:-0.95}"
export APPTAINERENV_OPENPI_DATA_HOME=/openpi-cache
export APPTAINERENV_XDG_CACHE_HOME=/cache/xdg APPTAINERENV_HF_HOME=/cache/hf APPTAINERENV_CUDA_CACHE_PATH=/cache/cuda
export APPTAINERENV_WANDB_MODE="${EXPO_WANDB_MODE:-disabled}" APPTAINERENV_WANDB_DIR=/output/wandb
# Pass credentials to the process only when supplied externally; never record their values.
if [[ -n "${WANDB_API_KEY:-}" ]]; then export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"; fi
if [[ -n "${WANDB_RUN_GROUP:-}" ]]; then export APPTAINERENV_WANDB_RUN_GROUP="$WANDB_RUN_GROUP"; fi
export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" APPTAINERENV_OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" APPTAINERENV_FASTVLA_GATHER_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export APPTAINERENV_EXPO_BATCH_SIZES="${EXPO_BATCH_SIZES:-8 64}" APPTAINERENV_EXPO_UPDATES="${EXPO_UPDATES:-20}" APPTAINERENV_EXPO_UTD="${EXPO_UTD:-20}" APPTAINERENV_EXPO_FSDP_DEVICES="${EXPO_FSDP_DEVICES:-1}" APPTAINERENV_EXPO_PHASE_TIMEOUT="${EXPO_PHASE_TIMEOUT:-100m}"
export APPTAINERENV_PYTHONPATH=/source:/opt/expo-ft/expo_ft/agents/vla/openpi/src:/opt/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client/src
container=(apptainer exec --nv --cleanenv --containall --no-mount cwd
  --home "$STAGE/home:/home/kevinon" --workdir "$STAGE/runtime" --pwd /source
  --bind "$STAGE/source:/source:ro,$STAGE/tools:/tools:ro,$STAGE/openpi-cache:/openpi-cache,$STAGE/demo:/demo:ro,$OUTPUT:/output,$STAGE/cache:/cache,$STAGE/jax-cache:/output/jax-cache"
  "$STAGE/learner.sif")
"${container[@]}" python -c 'from pathlib import Path; import expo_ft, train_pi_robo; import openpi.training.sharding as sharding; assert Path(expo_ft.__file__).is_relative_to("/source"); assert Path(train_pi_robo.__file__).is_relative_to("/source"); print("EXPO source:", expo_ft.__file__, train_pi_robo.__file__); print("OpenPI:", sharding.__file__); import jax, jax.numpy as j; assert jax.default_backend()=="gpu"; print(jax.__version__,jax.devices()); x=(j.ones((32,32))@j.ones((32,32))).block_until_ready(); assert float(x[0,0])==32' > "$OUTPUT/gpu-preflight.txt" 2>&1
"${container[@]}" python /tools/verify_assets.py /openpi-cache > "$OUTPUT/assets-verified.txt" 2>&1
"${container[@]}" cat /opt/expo-ft/docker/learner/requirements-installed.txt > "$OUTPUT/requirements-installed.txt"
if [[ "$1" == preflight ]]; then exit 0; fi
if [[ "$1" == validate ]]; then shift; set -- bash /tools/validate.sh "$@"; fi
(while true; do nvidia-smi --query-gpu=timestamp,uuid,memory.used,utilization.gpu --format=csv,noheader; sleep 5; done) > "$OUTPUT/gpu-memory.csv" &
monitor_pid=$!
"${container[@]}" "$@" &
child_pid=$!
set +e
wait "$child_pid"
rc=$?
child_pid=''
set -e
exit "$rc"
