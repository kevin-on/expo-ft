#!/usr/bin/env bash

source client/.venv/bin/activate

# REQUIRED: hostname or IP of the learner/eval node -- the machine running
# run_server.sh / run_server_async.sh / eval_policy.sh. The client dials it
# directly (no SSH tunnel), so it must be reachable from the robot machine.
# Override without editing this file: SERVER_HOST=my-gpu-box bash $0
SERVER_HOST="${SERVER_HOST:-}"

if [[ -z "$SERVER_HOST" ]]; then
    echo "ERROR: set SERVER_HOST to the learner's hostname (see the comment in $0)." >&2
    echo "       e.g. SERVER_HOST=my-gpu-box bash $0" >&2
    exit 1
fi

python -m client.run_client \
    --host="$SERVER_HOST" \
    --port=8102 \
    --config_task_path=configs/task/pick.py
