#!/usr/bin/env bash
# Run on a host that can access the learner's persistent checkpoint directory.
set -euo pipefail

if [[ $# != 1 || ! -d "$1" ]]; then
    echo "Usage: bash $0 EXISTING_CHECKPOINT_DIRECTORY" >&2
    exit 1
fi
CHECKPOINT_DIR=$(cd -- "$1" && pwd -P)
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo 'Press Enter to request a save after the current round updates; Ctrl+C exits only this script.'
echo 'Watch the learner log for: Manual checkpoint saved ...'
while IFS= read -r; do
    touch -- "$CHECKPOINT_DIR/save.request"
    echo 'Save requested (not yet confirmed complete). Press Enter to request again.'
done
