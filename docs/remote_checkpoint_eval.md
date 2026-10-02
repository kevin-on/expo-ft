# RAM checkpoint evaluation

Run from the `multi-robot` source and its matching OpenPI checkout
(the checkpoint-owned config implementation, currently OpenPI `590fa99`).
This adds an eval export/receiver; it does not change training checkpoint writes.
Do not merge or substitute the older `checkpoint-config` eval implementation.

New exports are saved as `checkpoints/<step>/trainable_weights.bin`, with unchanged
binary contents and common `assets/` / `model_config/` files. Existing
`checkpoints/<step>/eval/weights.bin` files remain readable without conversion.
The sender/receiver `--weights` argument accepts either file or the checkpoint
directory (new filename preferred). SFT packets can also initialize online FT
with a local frozen base; see [model configuration](model_config.md).

## Modes and storage

The server starts with local `--base-params` in **CPU RAM**, no GPU model.
Reception starts **Locked**. Press `R` to allow **one** transfer; press it again
before a sender connects to withdraw permission. The TUI displays Locked,
Ready for ONE transfer, Transferring, or Blocked during saving/eval. A sender
is rejected while locked, before exporting model data. Admission consumes the
permission: success, failure, cancellation, and timeout all leave reception locked.
Starting save/eval also clears unused permission. To intentionally replace the
current RAM checkpoint, press `R` again; this can discard unsaved RAM weights,
but never deletes an existing disk copy.

When armed, a sender reserves the receiver, exports a checkpoint,
then transfers a sealed RAM buffer through the existing transport. A failed
export or validation retains the previous checkpoint. There is no disk payload
staging and no queue of future checkpoints.

`S` saves (or verifies an identical existing copy of) the weights and common
configuration files. `E` is disabled until this checkpoint is `Saved`; it never
saves automatically. After saving, `E` prompts for the episode count **per robot**.
Type a positive integer and press Enter to start a separate GPU process to build
a fresh model and warm up. Empty Enter uses the displayed default (initially
`--episodes`, then the last chosen count); Backspace edits and Esc cancels without
loading a model. Reception is locked while choosing the count.
Save failure retains RAM and keeps eval disabled.
The TUI remains responsive during saving; offers are rejected during saving,
model loading, eval, and shutdown.

Once this RAM checkpoint is `Saved`, subsequent `S` skips disk reads, hash
comparison and writes. `E` also trusts that saved state without repeating disk
verification. Every newly accepted checkpoint resets this state, even if its
destination path is unchanged. This session-local state assumes saved files are
not externally removed or modified. Loading `--weights` directly from its canonical
`<experiments-root>/<checkpoint_path>/trainable_weights.bin` starts as `Saved`, enabling
`E` immediately without saving or re-reading the file for comparison. A file loaded
from elsewhere still requires `S` to persist it at the path recorded for eval.
`0`, `1`, and Space start a ready robot or both ready robots. They do not queue
starts while a robot is running/resetting. During eval, `r` / `t` repeat reset for
READY robot0 / robot1, returning to READY without starting or recording an episode.
The other robot can continue; busy robots ignore reset. Outside eval, `R` is still
the receive-permission key. Esc ends evaluation. The worker closes
its own robot connections and exits; only then can `R` allow another
checkpoint. The process boundary releases the CUDA context and model allocations.
The CPU base and received checkpoint remain available for reevaluation or saving.
The existing WS clients must connect to the eval worker's ports; preparing the
receiver does not start them, access cameras, or reset robots.

`S` saves in receive mode, with `RAM only`, `Saving`, `Saved`, or `Save failed`
and an explicit path. Receipt alone does not imply persistence. `Loaded from disk`
identifies a local load from outside its canonical destination; a new receipt says `RAM only` until saving verifies or
writes its disk copy. Save failures retain RAM. A different checkpoint at an
existing run/step is rejected, never silently overwritten.

```
<experiments-root>/
  <checkpoint_path>/               # sender's relative path, unchanged
    assets/config.json             # source SFT recipe, unchanged
    assets/<asset-id>/norm_stats.json
    model_config/config.json       # online only, unchanged from learner
    trainable_weights.bin               # required before every eval
  eval/<eval-id>/
    record.json
    config/eval_config.json
    config/robots/robot-0.json      # exact selected robot configuration(s)
    episodes.jsonl
    cube-records/
    logs/eval.log
```

The source of truth is `/iliad/u/kevinon/experiments/expo-ft/`, the server's default
`--experiments-root`. The sender supplies `training_run_id` and `checkpoint_path`
verbatim; the receiver does not reconstruct the path from kind, ID or step. Only
root-relative paths that stay within the root are accepted, including through
existing symlinks. Other checkpoint contents remain untouched.

Every eval gets a fresh directory and the existing four-field record schema:

```json
{
  "run_id": "<eval-id>",
  "training_run_id": "<registered-training-run-id>",
  "checkpoint_path": "online/<original-directory>/checkpoints/12430",
  "config_path": "eval/<eval-id>/config/eval_config.json"
}
```

All record paths are experiments-root-relative. No calculated counts, success
rates or checkpoint lists are stored in this record; derive them from
`episodes.jsonl`. There is no separate `summary.json`. An eval-only weight file
does not indicate that a full training/resume checkpoint has been collected.
WS videos remain on the WS under the specified video root and eval ID.
No optimizer, replay, temperature, training target actor, or frozen base is saved
in `trainable_weights.bin`. Online exports contain rollout actor, batch encoder, edit actor,
and **target** critic; SFT exports contain the inference actor. Array names,
shape/dtype/offsets, format version and frozen-base XXH3-128 are inside the file.
Common file text travels in the envelope so saved weights can also be re-sent by
themselves; when saved, those small files are materialized without rewriting them.
There is no separate manifest. All new weight/base content checks use XXH3-128.
Existing learner config files can contain legacy SHA-256 normalization fields;
those files remain unchanged and are checked by their legacy loader. SSH/TLS
security and transport's internal message identifiers are unchanged.

## Start transport once, reuse the relay

Use the existing SSH relay; do not launch or restart it from this tool. The latest
relay supports manual add/remove, **no automatic SSH reconnect**, and transport
telemetry. Preserve `relay_state_file`, `stats_file`, TLS/token paths, peers, and
chunk settings in transport configs. Follow [the split operating guide](../scripts/split/README.md).
The new application requires this revision of both transport sidecars (`next`
and `progress` local operations). Do not attach to sidecars used by a running
training job. Reusing SSH tunnels does not mean sharing a live training mailbox.

Keep the authentication agent and relay in a separate tmux session (for example,
`links`) so eval and online FT can reuse them when compute hosts and ports match.
The RAM transport has a separate lifetime: if either sidecar restarts, the peer
rejects the changed session epoch. Stop both applications and sidecars, use fresh
mailbox directories on both sides, then restart them together. Retrying the sender
command alone does not clear this failure. Launchers that stop their sidecar on
exit require this procedure even when only the receiver TUI was quit. Preserve the
SSH relay and agent during this restart.

On both compute machines start a transport with its normal matching config and a
**fresh private mailbox**. Keep the two sidecars alive across multiple sends:

```bash
python -m expo_ft.distributed.transport --config /path/to/transport.json
```

ILIAD compute, interactive terminal, expose one GPU:

```bash
python -m expo_ft.eval.server \
  --mailbox /path/to/inference-mailbox \
  --base-params /local/pi05_base/params \
  --experiments-root /iliad/u/kevinon/experiments/expo-ft \
  --client-video-dir /scr/kevinon/workspace/expo-ft-fork/data/videos \
  --robots 0 1 --episodes 30
```

The optional `--weights /.../trainable_weights.bin` loads a saved packet into RAM.
`--platform cpu` is for isolated testing only. Do not run actual models on the robot WS.

DeltaAI compute, sender (CPU-only, same runtime/source and existing sender mailbox):

```bash
python -m expo_ft.eval.send \
  --mailbox /path/to/learner-mailbox \
  --experiments-root /local/experiments/expo-ft --kind online \
  --checkpoint-path online/my-training-run/checkpoints/12430 \
  --training-run-id my-training-run \
  --initial-sft-checkpoint /local/initial-sft/<step>

python -m expo_ft.eval.send \
  --mailbox /path/to/learner-mailbox \
  --experiments-root /local/experiments/expo-ft --kind sft \
  --checkpoint-path sft/my-sft-run/checkpoints/4999 \
  --training-run-id my-sft-run
```

The source is `<experiments-root>/<checkpoint-path>` and exactly the same relative
path is sent to ILIAD. If source files have not yet moved into the shared layout,
use `--checkpoint /old/absolute/checkpoint/path` instead of `--experiments-root`,
still specifying the registered `--checkpoint-path` and `--training-run-id`.
The old absolute location is never used as a destination or placed in the eval record.
Re-sending `--weights` preserves its embedded path/ID and does not allow overrides.
Prototype `EXPOEV01` packets predate relative-path metadata; re-export those rather
than guessing a new destination. The current envelope is `EXPOEV02`.
SFT `--replan-steps` defaults to 8; online uses its checkpoint-owned value.
`--weights /.../trainable_weights.bin` re-sends an existing export instead of extracting
again. Sender checks receiver admission before reading model data. Finite transfer
reservations prevent a dead sender from blocking eval forever; expired sends must
be retried, not automatically installed. See sidecar quotas/timeouts for large
checkpoints. The sender currently reads the full online Orbax agent onto CPU to
extract the target critic, then discards training state; this never sends optimizer
bytes over the relay but still incurs source-disk read time.

## Config compatibility

SFT requires verified v2 config metadata. Old v1 recipes must first be explicitly
verified against their recorded OpenPI source with `scripts/migrate_sft_config.py`.
Old online checkpoints without `model_config` similarly need their original
resolved training settings; the exporter does not guess them. Migration is
separate from export, which does not modify the original checkpoint.

## Validation

Device-free: `python -m pytest --noconftest -q -p no:cacheprovider tests/test_remote_eval.py tests/test_robot_eval.py`.
On allocated compute only, `tests/gpu/remote_eval.py` provides separate CPU export,
GPU reference and GPU compact-model parity stages using recorded observations.
Never run WS clients or real robot environments as a transfer/model test.
