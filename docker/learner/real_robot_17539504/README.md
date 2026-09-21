# Real-robot run, 2026-09-21

This is a job-specific launch snapshot, not a portable cluster default.
The existing 12-hour sleep allocation is **17539504** on **iris6**, account
iris / partition iris-hi, A40 x2, 8 CPUs, 64 GiB RAM. Scheduled end was
2026-09-21 21:50:41 PDT; recheck Slurm before reuse. Do not cancel the holding job
just to stop the learner step.

Use Slurm commands on sc-codex, reached through scdt:

```bash
srun --jobid=17539504 --overlap -n1 -c1 --gres=gpu:a40:2 nvidia-smi
srun --jobid=17539504 --overlap -n1 -c8 --gres=gpu:a40:2 \
  bash /iris/u/kevinon/workspace/expo-ft-real/17539504/learner.sh
```

The workstation tmux session `expo-train` has `learner`, `robot0`, and `robot1`
windows. The learner window runs the SSH/Slurm command. Rollout windows are
initially idle. Both NUC servers run in the existing `kevin:0` split window:

| NUC pane | Robot | RPC | Arm controller | Gripper |
| --- | --- | --- | --- | --- |
| top `%0` | 172.16.0.2 | 4242 | 50055 | 50054 / DA6UJOT5 |
| bottom `%1` | 172.16.0.3 | 4243 | 50051 | 50052 / DA6UJXZ9 |

DROID source: `/home/iliad/kevin/droid-fork`, Conda `expoft`. Server startup
only opens RPC listeners. Controller initialization/reset happens when rollout
clients service the learner's environment requests. The existing controller on
50053 is unrelated and was preserved. Both Franka IPs and both DROID TCP ports
responded from the expected hosts.

The workstation can connect directly to iris6:8102 and :8103; the temporary
TCP test passed on both ports. No SSH port forwarding is needed. With all four
camera serials configured, run in separate workstation terminals:

```bash
bash scripts/multi_robot/run_workstation_rollout.sh 0
bash scripts/multi_robot/run_workstation_rollout.sh 1
```

These commands initialize real hardware once connected. The wrapper requires
real, available, disjoint camera mappings and the configured SpaceMouse paths.
Robot 1's wrist is configured as `12841040` in `configs/robots/robot-1.json`
and `blank_camera_serials` is empty. User selected the existing pick.py reset
joints/bounds for both robots. SpaceMouse paths remain robot0=/dev/hidraw2,
robot1=/dev/hidraw0; recheck after USB changes.

## Learner and tracking

- User selected pi05_base plus freshly initialized LoRA. Initial replay and
  normalization use the existing 226-transition teleop recording (not validated
  successful demonstrations); this provenance must remain explicit.
- Two synchronous robots, global batch 8, UTD 20, FSDP 2, num_updates=3 per round,
  replan 8, delay 0, seed 42. Ten completed episodes across both robots precede
  updates. max_steps=10000 limits replay/run size for this first session.
- Model and replay checkpointing enabled, interval 1000 aggregate transitions.
- WandB entity/project: `kevinon-stanford-university/mtexpo`.
- Group: `pick-pi05base-r2-20260921`.
- Run name: `pick-r2-a40x2-b8-utd20-s42-j17539504`.
- Run ID: `znr5aqj0`; https://wandb.ai/kevinon-stanford-university/mtexpo/runs/znr5aqj0
- Authentication is read from a mode-600 credential file outside the repository.
  The scripts contain only its path, never the key. Do not enable shell xtrace
  around credential loading or dump process environments.

The learner entry point creates its checkpoint directory before calling the
checkpoint manager, so the launcher passes `--resume`. With no checkpoints it
starts fresh; existing model/replay state is preserved. WandB uses the same run
ID across setup retries. Restarting this exact script continues this experiment;
use a new run ID/name and output directory for a different experiment.

## Storage and logs

- Runtime stage: `/tmp/kevinon/expo-real-17539504` on iris6 local NVMe.
- Container and dependencies: previously validated SIF, copied locally. Current
  fork Python sources are overlaid on a local copy of its source tree; nested
  OpenPI remains pinned. `source-manifest.json` records file hashes.
- OpenPI cache is a writable node-local copy: its tokenizer helper calls chmod
  even when the tokenizer file already exists. Shared model assets are unchanged.
- Learner log: `<stage>/output/learner.log`.
- Node output is mounted in-container at `/scr/kevinon/data/expo-real-17539504`.
  This same absolute directory exists independently on the workstation so client
  video paths resolve to workstation NVMe, not `/output` at the workstation root.
- Actual camera videos stay on the workstation under that directory; they are
  not part of the compute-node checkpoint archive.
- Shared results: `/iris/u/kevinon/outputs/expo-ft-real/17539504`.
- Every ten minutes, committed replay files and the latest completed numeric
  model checkpoint are bundled into `recovery.tar` with an atomic replacement.
  `recovery.json` records the model step and replay prefix. Before any model
  checkpoint, this archive preserves replay but cannot restore trained weights.
- On learner exit, save recovery plus logs/WandB local files. Sudden node failure
  may lose work since the last recovery snapshot. All hot small-file writes stay
  local while the run is active.

Initial setup retries hit the checkpoint-directory flag omission and then a
read-only cache mount. Both are corrected in this launch snapshot; they occurred
before robot clients connected.

Final readiness check: learner initialized successfully and listened on both
8102/8103 at 2026-09-21 10:32:59 PDT. WandB API confirmed state `running` and
the configured group/name plus batch=8, UTD=20, robots=2, updates=3, FSDP=2,
seed=42. No rollout client had connected and no robot reset was initiated.
Camera connection was still in progress; the latest enumeration contained
robot 0 side 38651013 and wrist 15577469. An optional check using the stored
NUC sudo credential was rejected by automatic approval review and not executed;
controller-launch authentication remains unverified in this session.

## End-to-end attempt at 11:12–11:18 PDT

The existing tmux robot shells lacked supplementary `zed` membership. After user
approval, `newgrp zed` refreshed each shell. No filesystem permissions changed.
Concurrent SDK enumeration failed transiently; the rollout preflight and DROID
MultiCameraWrapper now share a per-user file lock across discovery/opening.
The wrapper checks availability only for its own robot, while validating both
robots' ownership and SpaceMouse mappings. Canonical and nested DROID copies
were updated; the five existing camera regression cases passed in the existing
`/scr/kevinon/tmp/droid-integration-test-venv` environment.

On the retry, robot0 environment creation succeeded at 11:16:06 and robot1 at
11:16:54. All four cameras opened HD1080@15 with depth NONE; reset requests were
issued for both, and robot0 logged a policy step at 11:17:23. This is not a
completed training validation: no complete round or gradient update was observed.

NUC logs reported communication_constraints_violation on robot0 at 11:16:43
(success rate 0.693, 23 consecutive packets lost), and robot1 at 11:18:04
(success rate 0.38). The controllers automatically recovered. Both rollout
clients were stopped with Ctrl+C; the learner then exited on client disconnect.
The 12-hour GPU allocation and NUC servers/controllers were preserved. The logs
window remains a tail process, not an active learner. Real-time control threads
were already SCHED_FIFO (robot client priority 99, server priority 80); the cause
of missed control deadlines is not yet established. Do not call this a GPU OOM
or a proven physical network fault. Controller logs are append-only and contain
older failures too; distinguish the timestamps above.

## Robot1-only comparison at 11:37–11:39 PDT

`learner_robot1.sh` uses num_robot=1 and port 8103, retaining batch8/UTD20/
FSDP2/num_updates3. It selects the existing single-robot training implementation,
so this also changes the orchestration path relative to the two-robot round loop.
Run: pick-robot1-a40x2-b8-utd20-s42-j17539504; WandB ID 66d7e522;
group pick-pi05base-r1-20260921. Shared results suffix is 17539504-robot1.
A dedicated snapshot_robot1.py handles the single-robot replay directory.

Robot0 DROID server and its owned arm/gripper controllers were stopped before
starting this test; the unrelated pre-existing 50053 server was preserved.
Robot1 cameras opened HD1080@15/NONE and environment/reset completed at 11:37:54.
The run produced episode videos but was stopped before learner warmup completed.
NUC robot1 log showed measured-torque safe-range violations at 11:38:17.856
(success rate 1.0) and 11:38:47.296 (0.97), plus communication violations at
11:38:33.031 (47 consecutive missed packets, success rate 0.53) and 11:39:00.285
(46, 0.5346). Thus one active robot still reproduces the communication problem;
there is also a separate torque-sensor-range fault. No claim of a successful
end-to-end training test is warranted.

Both owned NUC servers/controllers and both workstation clients are now stopped.
The single learner waited to reconnect after its client exited (unlike the
multi-robot fail-fast path), so its specific Slurm step 17539504.25 was sent TERM.
The original GPU allocation remains; logs-r1 is only a log viewer.
