#!/usr/bin/env bash
#TPUQ --pool=usc2
#TPUQ --bucket=gs://soe-iris-kevin-usc2
#TPUQ --chips=4
set -euo pipefail

repo_root="$(pwd)"
export EXPO_SMOKE_OUTPUT="${repo_root}/logs/tpu-smoke"
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=tpu WANDB_MODE=disabled
export OPENPI_DATA_HOME=/opt/tpuq/expo-openpi-cache
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$EXPO_SMOKE_OUTPUT" "$OPENPI_DATA_HOME"

# Pin the same OpenPI checkout used for the local multi-robot work.
openpi_dir="${repo_root}/expo_ft/agents/vla/openpi"
git init "$openpi_dir"
git -C "$openpi_dir" fetch --depth=1 https://github.com/pd-perry/openpi.git 46407a41183b037313a383ff679683f2773b766d
git -C "$openpi_dir" checkout --detach FETCH_HEAD

uv_bin="$HOME/.local/bin/uv"
runtime_dir=/opt/tpuq/expo-smoke-py311-jax053
(
    flock -x 9
    "$uv_bin" venv --python 3.11 --allow-existing "$runtime_dir"
    "$uv_bin" pip install --python "$runtime_dir/bin/python" 'torch==2.7.1' \
        --index-url https://download.pytorch.org/whl/cpu
    "$uv_bin" pip install --python "$runtime_dir/bin/python" \
        -r tests/tpu/requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
) 9>/opt/tpuq/expo-smoke-runtime.lock
export PATH="$runtime_dir/bin:$PATH"
export PYTHONPATH="$repo_root:$openpi_dir/src:$openpi_dir/packages/openpi-client/src"

python -m pip freeze > "$EXPO_SMOKE_OUTPUT/requirements-installed.txt" 2>/dev/null || \
    "$uv_bin" pip freeze --python "$runtime_dir/bin/python" > "$EXPO_SMOKE_OUTPUT/requirements-installed.txt"

save_results() {
    result=$?
    printf '%s\n' "$result" > "$EXPO_SMOKE_OUTPUT/exit-code.txt"
    # Keep small diagnostic artifacts; checkpoint and synthetic images stay on the worker.
    gcloud storage cp "$EXPO_SMOKE_OUTPUT"/*.json "$EXPO_SMOKE_OUTPUT"/*.txt \
        "${TPUQ_BUCKET}/expo-ft/smoke/${TPUQ_TASK_ID}/" || true
    exit "$result"
}
trap save_results EXIT

JAX_PLATFORMS=cpu python -m pytest -q tests/cpu/test_robot_round.py \
    2>&1 | tee "$EXPO_SMOKE_OUTPUT/cpu-round-tests.txt"
timeout 45m python tests/tpu/smoke.py 2>&1 | tee "$EXPO_SMOKE_OUTPUT/run.txt"
