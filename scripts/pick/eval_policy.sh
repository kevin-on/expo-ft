#!/usr/bin/env bash
set -euo pipefail
export EXPO_EVAL_KIND=online
export EXPO_EVAL_CHECKPOINT=${ONLINE_CHECKPOINT:?Set a completed online checkpoint step}
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/eval_sft_policy.sh" "${1:-both}"
