#!/usr/bin/env bash
# WS launcher. Starts real hardware when the eval server requests an environment.
set -euo pipefail
robot_index=${1:?Usage: run_sft_eval_client.sh 0|1}
case "$robot_index" in 0|1) ;; *) echo 'Robot index must be 0 or 1' >&2; exit 2;; esac
: "${EXPO_EVAL_HOST:?Set the GPU eval host or local SSH tunnel endpoint}"
source /scr/kevinon/env.sh
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
exec client/.venv/bin/python -m client.run_client \
    --host "$EXPO_EVAL_HOST" --port "$(( ${EXPO_EVAL_BASE_PORT:-8202} + robot_index ))" \
    --config-task-path configs/task/pick.py \
    --robot-config "configs/robots/robot-${robot_index}-sft-eval.json"
