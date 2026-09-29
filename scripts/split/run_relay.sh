#!/usr/bin/env bash
# On the DeltaAI login node. Does not start models or robot clients.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_profile "$@"
[[ $(hostname -s) == "$DELTA_LOGIN_NODE" ]] || {
    echo "Run on $DELTA_LOGIN_NODE (configured relay address $DELTA_LOGIN_IP)" >&2; exit 1;
}
require_file "$DELTA_RUN/link/relay.json"
[[ -S "$AGENT_SOCKET" ]] || { echo 'Start the dedicated WS agent forward first' >&2; exit 1; }
timeout --kill-after=5s 30s ssh -F "$DELTA_RUN/link/ssh-config" \
    -o ConnectTimeout=10 -o ConnectionAttempts=1 iliad-bench hostname
cd "$DELTA_SHARED_SOURCE"
exec python3 -u -m expo_ft.distributed.relay --config "$DELTA_RUN/link/relay.json" \
    --log-file "$DELTA_RUN/relay.log"
