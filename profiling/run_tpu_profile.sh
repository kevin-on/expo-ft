#!/usr/bin/env bash
#TPUQ --pool=usc2
#TPUQ --bucket=gs://soe-iris-kevin-usc2
#TPUQ --chips=4
set -euo pipefail
repo_root="$PWD"
: "${TPUQ_TASK_ID:?}" "${TPUQ_RUN_NUMBER:?}" "${TPUQ_BUCKET:?}"
[[ "$TPUQ_BUCKET" == gs://soe-iris-kevin-usc2 ]]
[[ "$TPUQ_POOL" == usc2 ]]
[[ "$(awk -F, '{print NF}' <<< "$TPUQ_CHIP_IDS")" == 4 ]]
stage=$(mktemp -d "/opt/tpuq/expo-profile-${TPUQ_TASK_ID}-${TPUQ_RUN_NUMBER}-XXXXXX")
output="$stage/output"
mkdir -p "$output"
remote_output="${TPUQ_BUCKET}/expo-ft/learner-profile/${TPUQ_TASK_ID}/attempt-${TPUQ_RUN_NUMBER}"
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=tpu WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 FASTVLA_GATHER_THREADS=8
export XDG_CACHE_HOME="$stage/cache" HF_HOME="$stage/hf" TMPDIR="$stage/tmp"
mkdir -p "$TMPDIR"
# Final artifacts stay in the same region. Never upload compilation caches.
publish() {
    gcloud storage rsync --recursive --exclude='(^|/)(jax-cache|checkpoint)(/|$)' \
        "$output" "$remote_output"
}
finish() {
    rc=$?
    trap - EXIT TERM INT
    printf '%s\n' "$rc" > "$output/exit-code.txt"
    if ! publish; then
        echo 'Failed to preserve validation outputs in GCS' >&2
        if (( rc == 0 )); then rc=74; fi
    fi
    echo "TPU results: $remote_output (exit=$rc)"
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'commit=%s\ntask=%s\nattempt=%s\nchips=%s\n' \
    "$TPUQ_COMMIT_SHA" "$TPUQ_TASK_ID" "$TPUQ_RUN_NUMBER" "$TPUQ_CHIP_IDS" > "$output/source.txt"
df -h "$stage"
(( $(df -B1 --output=avail "$stage" | tail -n 1) >= 60*1024*1024*1024 ))
# Use the same public weights and recorded episode as the A40 profile.
gcloud storage cp "${TPUQ_BUCKET}/expo-ft/assets/validation-20260921.tar" "$stage/assets.tar"
tar -xf "$stage/assets.tar" -C "$stage"
rm "$stage/assets.tar"
export OPENPI_DATA_HOME="$stage/openpi-cache"
python3 profiling/verify_tpu_assets.py "$stage" > "$output/assets-verified.json"
uv_bin=$(command -v uv || printf '%s' "$HOME/.local/bin/uv")
# Separate from other experiments' environments; pin the known TPU JAX stack.
runtime_dir=/opt/tpuq/expo-profile-py311-jax053-v1
(
    flock -x 9
    "$uv_bin" venv --python 3.11 --allow-existing "$runtime_dir"
    "$uv_bin" pip install --python "$runtime_dir/bin/python" 'torch==2.7.1' \
        --index-url https://download.pytorch.org/whl/cpu
    "$uv_bin" pip install --python "$runtime_dir/bin/python" -r tests/tpu/requirements.txt \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    "$uv_bin" pip check --python "$runtime_dir/bin/python"
) 9>/opt/tpuq/expo-profile-runtime.lock
export PATH="$runtime_dir/bin:$PATH"
"$uv_bin" pip freeze --python "$runtime_dir/bin/python" > "$output/requirements-installed.txt"
openpi_dir="$repo_root/expo_ft/agents/vla/openpi"
git init "$openpi_dir"
git -C "$openpi_dir" fetch --depth=1 https://github.com/pd-perry/openpi.git 46407a41183b037313a383ff679683f2773b766d
git -C "$openpi_dir" checkout --detach FETCH_HEAD
export PYTHONPATH="$repo_root:$openpi_dir/src:$openpi_dir/packages/openpi-client/src"
python scripts/multi_robot/setup_droid.py
python - <<'PY' > "$output/tpu-preflight.txt"
import jax
import jax.numpy as jnp
assert jax.default_backend() == 'tpu'
assert jax.device_count() == jax.local_device_count() == 4, jax.devices()
print(jax.__version__, jax.devices())
print((jnp.ones((32,32)) @ jnp.ones((32,32))).block_until_ready())
PY
# Freeze both variants before editing this task's private checkout.
base=f92c32a0d1eae3aca2282ff1787228c60b55a9be
git cat-file -e "$base^{commit}" 2>/dev/null || git fetch --depth=1 origin "$base"
python profiling/prepare_snapshot.py "$stage/snapshot"
sha256sum "$stage/snapshot"/*.py > "$output/source-sha256.txt"
for variant in baseline baseline-repeat fixed; do
    source_variant=$variant
    extra_args=()
    if [[ "$variant" == baseline-repeat ]]; then
        source_variant=baseline
        extra_args=(--compilation-cache "$output/baseline/jax-cache")
    fi
    cp "$stage/snapshot/${source_variant}_expo_ft.py" expo_ft/agents/alg/expo_ft.py
    cp "$stage/snapshot/${source_variant}_temperature.py" expo_ft/networks/temperature.py
    echo "Starting $variant: $(date -Is)"
    set +e
    timeout --signal=TERM --kill-after=30s 60m python tests/gpu/learner_smoke.py \
        --backend tpu --num-devices 4 --fsdp-devices 4 \
        --dataset "$stage/demo" --params "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params" \
        --output "$output/$variant" --batch-size 4 --utd-ratio 1 --updates 6 \
        --skip-checkpoint --profile-updates "${extra_args[@]}" > "$output/$variant.log" 2>&1
    code=$?
    set -e
    printf '%s exit=%s timestamp=%s\n' "$variant" "$code" "$(date -Is)" | tee -a "$output/run-status.txt"
    publish
    if (( code != 0 )); then tail -n 80 "$output/$variant.log"; exit "$code"; fi
done
python tests/gpu/compare_profile.py "$output/baseline" "$output/fixed" \
    --control "$output/baseline-repeat" > "$output/comparison.json"
cat "$output/comparison.json"
