#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --job-name=expo-update-profile
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
#SBATCH --chdir=/iris/u/kevinon/workspace/expo-ft-update-profile
#SBATCH --output=/iris/u/kevinon/outputs/expo-ft-update-profile/logs/%x-%j.out
#SBATCH --error=/iris/u/kevinon/outputs/expo-ft-update-profile/logs/%x-%j.err
set -euo pipefail
# Weights, SIF, runtime caches and a full checkpoint must fit on local storage.
local_root=${TMPDIR:-/tmp}
available_bytes=$(df -B1 --output=avail "$local_root" | tail -n 1)
available_inodes=$(df --output=iavail "$local_root" | tail -n 1)
if (( available_bytes < 60 * 1024 * 1024 * 1024 || available_inodes < 100000 )); then
    hostname
    df -h "$local_root"
    df -i "$local_root"
    echo 'Insufficient local scratch space: require 60 GiB and 100000 free inodes.' >&2
    exit 75
fi
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/expo-gpu-${SLURM_JOB_ID}-XXXXXX")
RESULTS=/iris/u/kevinon/outputs/expo-ft-update-profile/gpu-${SLURM_JOB_ID}
mkdir -p "$STAGE/output" "$STAGE/home" "$STAGE/runtime" "$RESULTS"
export APPTAINER_CACHEDIR="$STAGE/cache" APPTAINER_TMPDIR="$STAGE/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
monitor_pid=""
recover() {
    rc=$?
    trap - EXIT TERM INT
    if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
    # Checkpoint payloads stay local; preserve logs, normalization, and comparison results.
    tar -C "$STAGE/output" --exclude='./small/checkpoint' --exclude='./default/checkpoint' \
      --exclude='*/jax-cache' --exclude='./cache' --exclude='./hf' --exclude='./cuda-cache' \
      -czf "$RESULTS/results.tar.gz" . || true
    cp "$STAGE/output/run-status.txt" "$RESULTS/" 2>/dev/null || true
    printf '%s\n' "$rc" > "$RESULTS/exit-code.txt"
    printf 'Validation outputs: %s\n' "$RESULTS"
    exit "$rc"
}
trap recover EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
hostname
printf 'Local stage: %s\nCUDA_VISIBLE_DEVICES=%s\n' "$STAGE" "${CUDA_VISIBLE_DEVICES:-}"
df -h "$STAGE"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$STAGE/output/gpu.txt"
image_source=/iris/u/kevinon/artifacts/expo-ft/expo-ft-learner-jax053.sif
cache_source=/iliad/u/kevinon/artifacts/expo-ft/openpi-cache
if [[ -n "${EXPO_LOCAL_INPUT_ROOT:-}" && -f "$EXPO_LOCAL_INPUT_ROOT/openpi-cache/manifest.json" && -f "$EXPO_LOCAL_INPUT_ROOT/learner.sif" ]]; then
    image_source="$EXPO_LOCAL_INPUT_ROOT/learner.sif"
    cache_source="$EXPO_LOCAL_INPUT_ROOT/openpi-cache"
fi
cp --reflink=auto --no-preserve=mode,ownership "$image_source" "$STAGE/learner.sif"
mkdir -p "$STAGE/openpi-cache" "$STAGE/demo"
cp ./learner_smoke.py "$STAGE/learner_smoke.py"
cp ./baseline_expo_ft.py "$STAGE/expo_ft.py"
cp ./baseline_temperature.py "$STAGE/temperature.py"
cp ./compare_profile.py "$STAGE/compare_profile.py"
sha256sum ./*.py > "$STAGE/output/source-sha256.txt"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export APPTAINERENV_JAX_PLATFORMS=cuda
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
# This Slurm allocation owns one GPU; retain 2% headroom outside the JAX pool.
export APPTAINERENV_XLA_PYTHON_CLIENT_MEM_FRACTION=0.98
export APPTAINERENV_OPENPI_DATA_HOME=/openpi-cache
export APPTAINERENV_XDG_CACHE_HOME=/output/cache
export APPTAINERENV_HF_HOME=/output/hf
export APPTAINERENV_CUDA_CACHE_PATH=/output/cuda-cache
export APPTAINERENV_WANDB_DIR=/output/wandb
export APPTAINERENV_OMP_NUM_THREADS=8
export APPTAINERENV_OPENBLAS_NUM_THREADS=8
export APPTAINERENV_FASTVLA_GATHER_THREADS=8
container=(apptainer exec --nv --cleanenv --containall --no-mount cwd \
  --home "$STAGE/home:/home/kevinon" --workdir "$STAGE/runtime" --pwd /opt/expo-ft \
  --bind "$STAGE/openpi-cache:/openpi-cache,$STAGE/demo:/demo:ro,$STAGE/output:/output,$STAGE/learner_smoke.py:/opt/expo-ft/tests/gpu/learner_smoke.py:ro,$STAGE/expo_ft.py:/opt/expo-ft/expo_ft/agents/alg/expo_ft.py:ro,$STAGE/temperature.py:/opt/expo-ft/expo_ft/networks/temperature.py:ro,$STAGE/compare_profile.py:/opt/expo-ft/tests/gpu/compare_profile.py:ro" "$STAGE/learner.sif")
# Check CUDA before copying 12.44 GB: fail early on an unhealthy allocation.
"${container[@]}" python -c 'import jax; import jax.numpy as j; assert len(jax.devices()) == 1; assert jax.default_backend() == "gpu"; print(jax.devices()); print((j.ones((32,32)) @ j.ones((32,32))).block_until_ready().shape)' > "$STAGE/output/gpu-preflight.txt" 2>&1
cp -R --reflink=auto --no-preserve=mode,ownership,xattr "$cache_source/." "$STAGE/openpi-cache/"
cp -R --no-preserve=mode,ownership,xattr /iris/u/kevinon/artifacts/expo-ft/demo/. "$STAGE/demo/"
"${container[@]}" python docker/learner/verify_assets.py /openpi-cache > "$STAGE/output/assets-verified.txt" 2>&1
"${container[@]}" cat /opt/expo-ft/docker/learner/requirements-installed.txt > "$STAGE/output/requirements-installed.txt"
(while true; do nvidia-smi --query-gpu=timestamp,uuid,memory.used,utilization.gpu --format=csv,noheader; sleep 1; done) > "$STAGE/output/gpu-memory.csv" &
monitor_pid=$!
params=/openpi-cache/openpi-assets/checkpoints/pi05_base/params
run_case() {
    label=$1
    shift
    printf 'Starting %s at %s\n' "$label" "$(date -Is)"
    set +e
    timeout --signal=TERM --kill-after=30s 30m "${container[@]}" python tests/gpu/learner_smoke.py \
      --dataset /demo --params "$params" "$@" > "$STAGE/output/${label}.log" 2>&1
    code=$?
    set -e
    printf '%s exit=%s timestamp=%s\n' "$label" "$code" "$(date -Is)" | tee -a "$STAGE/output/run-status.txt"
    cp "$STAGE/output/run-status.txt" "$RESULTS/run-status.txt"
    return "$code"
}
# The two processes run serially on the same GPU, with separate cold JIT caches.
for variant in baseline fixed; do
    cp "./${variant}_expo_ft.py" "$STAGE/expo_ft.py"
    cp "./${variant}_temperature.py" "$STAGE/temperature.py"
    run_case "$variant" --output "/output/$variant" --batch-size 2 --utd-ratio 1 --updates 6 --skip-checkpoint --profile-updates
done
"${container[@]}" python tests/gpu/compare_profile.py /output/baseline /output/fixed > "$STAGE/output/comparison.json"
cat "$STAGE/output/comparison.json"
