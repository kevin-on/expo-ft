#!/usr/bin/env bash
# prepare: writes/copies only launch configuration; starts no remote services.
# agent: keeps a dedicated, single-key agent forward alive until Ctrl-C.
set -euo pipefail
ACTION=${1:?Usage: workstation.sh prepare|agent RUN [PROFILE]}
shift
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_profile "$@"
SSH=(ssh -F "$WS_SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=20)
ILIAD_SSH=("${SSH[@]}" -o ProxyJump=scdt -o ForwardAgent=no -o IdentitiesOnly=yes
    -i "$WS_SSH_KEY" -o StrictHostKeyChecking=yes
    -o UserKnownHostsFile="$KNOWN_HOSTS_ROOT/iliad-known-hosts" "kevinon@$ILIAD_HOST")
case "$ACTION" in
prepare)
    for file in "$KNOWN_HOSTS_ROOT/scdt-known-hosts" "$KNOWN_HOSTS_ROOT/iliad-known-hosts"; do require_file "$file"; done
    command -v openssl >/dev/null
    umask 077
    mkdir -p "$WS_LINK_ROOT"
    mkdir "$WS_LINK"
    mkdir "$WS_LINK/link" "$WS_LINK/launch"
    cp "$LAUNCH_DIR/"*.sh "$WS_LINK/launch/"
    cp "$PROFILE" "$WS_LINK/launch/deltaai_iliad.env"
    cp "$KNOWN_HOSTS_ROOT/scdt-known-hosts" "$KNOWN_HOSTS_ROOT/iliad-known-hosts" "$WS_LINK/link/"
    openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes \
        -keyout "$WS_LINK/link/key.pem" -out "$WS_LINK/link/cert.pem" \
        -days 3 -subj /CN=expo-link >"$WS_LINK/certificate.log" 2>&1
    openssl rand -hex 32 >"$WS_LINK/link/token"
    /usr/bin/python3 - "$WS_LINK/link" "$DELTA_RUN/link" "$AGENT_SOCKET" \
        "$DELTA_LOGIN_IP" "$DELTA_NODE_IP" "$ILIAD_HOST" "$CONNECTIONS" <<'PY'
import json
from pathlib import Path
import sys
local, remote, agent, login, compute, iliad, count = sys.argv[1:]
root = Path(local)
count = int(count)
if not 1 <= count <= 64:
    raise SystemExit('CONNECTIONS must be 1..64')
for role, listen, peers in (
    ('learner', ['0.0.0.0', 24101], [[login, 24200 + i] for i in range(count)]),
    ('inference', ['127.0.0.1', 24102], [['127.0.0.1', 24300 + i] for i in range(count)]),
):
    cfg = dict(mailbox='/mailbox', listen=listen, peers=peers,
               parallel_connections=count, record_connections=4,
               chunk_bytes=4 * 1024**2, token_file='/link/token',
               tls=dict(cert_file='/link/cert.pem', key_file='/link/key.pem', ca_file='/link/cert.pem'))
    (root / ('transport-' + role + '.json')).write_text(json.dumps(cfg, indent=2) + '\n')
relay = dict(ssh_config=remote + '/ssh-config', ssh_host='iliad-bench', connections=count,
             listen_host=login, first_port=24200, destination=['127.0.0.1', 24102],
             reverse=dict(listen_host='127.0.0.1', first_port=24300, destination=[compute, 24101]),
             output=remote + '/relay-endpoints.json')
(root / 'relay.json').write_text(json.dumps(relay, indent=2) + '\n')
(root / 'ssh-config').write_text('''Host *
  BatchMode yes
  Ciphers aes128-gcm@openssh.com
  Compression no
  ControlMaster no
  ControlPath none
  ForwardAgent no
  IdentityAgent "{agent}"
  IdentityFile none
  IdentitiesOnly no
  StrictHostKeyChecking yes
Host scdt-bench
  HostName scdt.stanford.edu
  User kevinon
  UserKnownHostsFile "{remote}/scdt-known-hosts"
Host iliad-bench
  HostName {iliad}
  User kevinon
  ProxyJump scdt-bench
  UserKnownHostsFile "{remote}/iliad-known-hosts"
'''.format(agent=agent, remote=remote, iliad=iliad))
PY
    # Only this run's small launch files/credentials; no model/dataset copies.
    # A pre-existing remote run is an error, never an overwrite.
    printf -v delta_cmd 'umask 077; mkdir -p -- %q && mkdir -- %q && tar -xf - -C %q' \
        "$DELTA_RUN_ROOT" "$DELTA_RUN" "$DELTA_RUN"
    tar -cf - -C "$WS_LINK" launch link | "${SSH[@]}" deltaai "$delta_cmd"
    printf -v iliad_cmd 'umask 077; mkdir -p -- %q && mkdir -- %q && tar -xf - -C %q' \
        "$ILIAD_RUN_ROOT" "$ILIAD_RUN" "$ILIAD_RUN"
    tar -cf - -C "$WS_LINK" launch link | "${ILIAD_SSH[@]}" "$iliad_cmd"
    printf 'Prepared, no services started.\nDeltaAI launch directory: %s/launch\nILIAD launch directory: %s/launch\n' "$DELTA_RUN" "$ILIAD_RUN"
    ;;
agent)
    require_file "$WS_LINK/link/token"
    [[ ! -e "$WS_LINK/agent.sock" ]] || { echo 'This run already has an agent socket' >&2; exit 1; }
    umask 077
    AGENT_PID=
    FORWARD_ADDED=false
    cleanup_agent() {
        trap - EXIT INT TERM
        if "$FORWARD_ADDED"; then
            "${SSH[@]}" -O cancel -R "$AGENT_SOCKET:$WS_LINK/agent.sock" deltaai || \
                echo "Could not remove remote forward $AGENT_SOCKET; check the DeltaAI master" >&2
        fi
        [[ -z "$AGENT_PID" ]] || kill "$AGENT_PID" 2>/dev/null || true
        [[ -z "$AGENT_PID" ]] || wait "$AGENT_PID" 2>/dev/null || true
    }
    trap cleanup_agent EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    ssh-agent -D -a "$WS_LINK/agent.sock" >"$WS_LINK/agent.log" 2>&1 &
    AGENT_PID=$!
    for ((i=0; i<50; i++)); do
        [[ ! -S "$WS_LINK/agent.sock" ]] || break
        kill -0 "$AGENT_PID"
        sleep .1
    done
    SSH_AUTH_SOCK="$WS_LINK/agent.sock" ssh-add "$WS_SSH_KEY"
    "${SSH[@]}" -O check deltaai
    "${SSH[@]}" -O forward -R "$AGENT_SOCKET:$WS_LINK/agent.sock" deltaai
    FORWARD_ADDED=true
    echo 'Dedicated Stanford-key agent forwarded. Keep this terminal open; Ctrl-C removes this forward/agent only.'
    wait "$AGENT_PID"
    ;;
*) echo 'Action must be prepare or agent' >&2; exit 2;;
esac
