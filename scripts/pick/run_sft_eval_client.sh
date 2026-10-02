#!/usr/bin/env bash
# Real hardware launcher. Controls live in the GPU terminal, not these clients.
set -euo pipefail
selection=${1:-both}
if (( $# )); then shift; fi
case "$selection" in both|0|1) ;; *) echo 'Usage: run_sft_eval_client.sh [both|0|1]' >&2; exit 2;; esac
: "${EXPO_EVAL_HOST:?Set the eval host or local SSH tunnel endpoint}"
source /scr/kevinon/env.sh
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
python=${EXPO_CLIENT_PYTHON:-client/.venv/bin/python}
args=(--host "$EXPO_EVAL_HOST" --config-task-path configs/task/pick.py "$@")
base_port=${EXPO_EVAL_BASE_PORT:-8202}
if [[ $selection != both ]]; then
    exec "$python" -m client.run_client "${args[@]}" --port "$((base_port+selection))" \
        --robot-config "configs/robots/robot-$selection.json"
fi
logs=${EXPO_CLIENT_LOG_DIR:-data/logs/eval-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$logs"
pids=()
cleanup() {
    trap - EXIT
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for robot in 0 1; do
    "$python" -m client.run_client "${args[@]}" --port "$((base_port+robot))" \
        --robot-config "configs/robots/robot-$robot.json" </dev/null >>"$logs/robot$robot.log" 2>&1 &
    pids+=("$!")
done
printf 'Clients started; logs: %s/robot{0,1}.log. Space/q: GPU terminal.\n' "$logs"
status=0
wait -n "${pids[@]}" || status=$?
exit "$status"
