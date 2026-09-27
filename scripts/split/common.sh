#!/usr/bin/env bash
# Sourced only. The profile contains paths/settings, never credentials.
load_profile() {
    RUN=${1:?Usage: SCRIPT RUN [PROFILE]}
    [[ "$RUN" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$ ]] || {
        echo 'RUN must be 1..48 letters, digits, underscores or hyphens' >&2; return 2;
    }
    LAUNCH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    PROFILE=${2:-$LAUNCH_DIR/deltaai_iliad.env}
    PROFILE=$(realpath "$PROFILE")
    source "$PROFILE"
    DELTA_RUN=$DELTA_RUN_ROOT/$RUN
    ILIAD_RUN=$ILIAD_RUN_ROOT/$RUN
    WS_LINK=$WS_LINK_ROOT/$RUN
    AGENT_SOCKET=/tmp/expo-split-$RUN.sock
}
require_file() { [[ -f "$1" ]] || { echo "Missing file: $1" >&2; return 1; }; }
