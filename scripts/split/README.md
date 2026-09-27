# DeltaAI learner + ILIAD inference launch

These scripts package the previously documented manual commands. They use the
prepared EXPO/OpenPI source and architecture-specific containers, **without**
changing model/training/transport code or starting a robot client automatically.
No service starts by copying these files.

## Profile and defaults

Edit `deltaai_iliad.env` before preparation (or pass a trusted alternative profile
as the final argument). It records the **2026-09-26 allocation's** job/node IDs,
IP addresses, node-local images/models/source and shared dataset/output paths.
Recheck allocation/step availability before launching; a replacement allocation
needs new paths/IPs and restaging. These scripts do not allocate GPUs or stage
large artifacts. If `scdt` is unavailable, fix connectivity before proceeding;
there is no fallback SSH/authentication route.

Defaults match the preceding manual recipe: mixed20 SFT step4999 base and its
normalization stats, canonical mixed20 HDF5 demos, 2 robots, batch 64, UTD 20,
3 updates/round, 10 warmup episodes per robot, 4 learner GPUs/FSDP1, 1 inference GPU, max 20000 total transitions,
checkpoint+replay every 2000 transitions, W&B `mtexpo` / group `split-mixed20-r2`.
The task remains `pick.py` (80-step episode limit). This is a **fresh online run**,
not a resume of earlier online optimizer/replay state.

Use the same RUN everywhere for output paths and the W&B name. For a coordinated
restart, set the same fresh `SPLIT_SESSION` in both deployed profiles (otherwise it
defaults to RUN). Stop both old model/transport roles first; relay/agent can stay up.
Each launch uses a fresh temporary IPC directory and holds a per-role lock.
A failed startup containing only `wandb_id.txt` is archived before restarting;
actual checkpoints or replay records are preserved and require explicit resume
configuration or a fresh RUN. Earlier role logs are retained with the Slurm step ID.
Preparing an existing run or starting a duplicate role remains an error.
Do not run with `bash -x`: launch-time environments contain credentials.

## 1. Workstation: prepare small launch files

```bash
cd /scr/kevinon/workspace/expo-ft-split
bash scripts/split/workstation.sh prepare split-mixed20-r2-20260926-01
```

Creates private token/TLS files (three-day certificate), the two transport JSONs,
relay SSH configuration and a copy of these launch scripts/profile. Copies only
these small files to each cluster's fresh shared run directory. It does not copy
SSH private keys, checkpoints or datasets, and starts no remote process.
The verified public known-host files are reused; unknown host keys are rejected.
A failure can leave partially prepared directories; they are not silently replaced.

Then, in a dedicated WS terminal:

```bash
bash scripts/split/workstation.sh agent split-mixed20-r2-20260926-01
```

This starts a dedicated agent containing only the Stanford key and adds one Unix
socket forward to the existing DeltaAI SSH master. Keep it running. Ctrl-C removes
only that forward and agent; it does not terminate the shared SSH master.
A working existing `ssh deltaai` multiplex connection is required.

## 2. DeltaAI login: relay

```bash
ssh -F /scr/kevinon/.ssh/config deltaai
cd /work/hdd/bgqe/kon/expo-ft/runs/split-mixed20-r2-20260926-01/launch
bash run_relay.sh split-mixed20-r2-20260926-01
```

Run on the profile's login node (`gh-login03`). It performs one read-only SSH
hostname check, then starts the existing 64-connection relay supervisor. Keep this
terminal/tmux pane open. Listen/forward ports are the existing 24101/24102,
24200–24263 and 24300–24363 convention; only one run may use these ports at a time.

## 3. DeltaAI login: learner (another pane)

```bash
cd /work/hdd/bgqe/kon/expo-ft/runs/split-mixed20-r2-20260926-01/launch
bash run_role.sh learner split-mixed20-r2-20260926-01
```

## 4. ILIAD login: inference

From WS, use the clean-shell route if normal login initialization is slow:

```bash
ssh -t -F /scr/kevinon/.ssh/config scdt 'cd /tmp && exec bash --noprofile --norc'
# On scdt (which does not have Slurm commands):
ssh -t sc-codex 'cd /tmp && exec bash --noprofile --norc'
# On sc-codex:
cd /iliad/u/kevinon/outputs/expo-ft/split-online/split-mixed20-r2-20260926-01/launch
bash run_role.sh inference split-mixed20-r2-20260926-01
```

Both role commands are run on **login nodes**, outside an existing compute shell.
They show current Slurm job/steps, start one exclusive resource step in the held
allocation, and start a transport sidecar plus the model inside the validated
container. No nested `srun`, `--overlap`, new allocation or parent-job cancellation.
Use login-node tmux for persistence, then reconnect to the same login node.
Node-local stage paths must still exist; missing inputs fail before model startup.
The inference process waits for robot clients; it does not open cameras itself.

Logs: shared run directory `learner.log` / `inference.log`, `transport-ROLE.log`,
and DeltaAI `relay.log`. Learner checkpoints: `DELTA_RUN_ROOT/RUN/checkpoints`.
Inference's video directory points to the **workstation** path in the profile.

## 5. Robot clients: existing launcher, only when ready for motion

After both model processes initialize and inference is waiting for clients:

```bash
cd /scr/kevinon/workspace/expo-ft-split
EXPO_LEARNER_HOST=iliad-hgx-1.stanford.edu EXPO_LEARNER_BASE_PORT=8102 \
  bash scripts/multi_robot/run_workstation_rollout.sh 0
# Another workstation pane, when robot1 is ready:
EXPO_LEARNER_HOST=iliad-hgx-1.stanford.edu EXPO_LEARNER_BASE_PORT=8102 \
  bash scripts/multi_robot/run_workstation_rollout.sh 1
```

These commands access actual devices and permit reset/motion. Check physical
readiness and current SpaceMouse/camera mappings before executing them.
The legacy environment name `EXPO_LEARNER_HOST` points to **inference** here.

## Stop

Stop robot clients first, then Ctrl-C the role commands; their exit handlers stop
only their own model/transport children. Ctrl-C relay, then Ctrl-C the WS agent
terminal. Parent held allocations remain allocated and charged. For an unattended
step, inspect `squeue --steps -j JOB` and terminate only its exact `JOB.STEP`.
Do not use `scancel JOB` unless returning the entire allocation is intended.
Credentials remain in each run's private `link/` directory for explicit cleanup;
no script recursively deletes outputs/checkpoints or cancels unrelated forwards.

For short validation, set `WARMUP_EPISODES=1` in the run profile. The first round
is warmup; updates can begin after the second round (subject to batch size).
This sets `--split_warmup_episodes`; the default remains 10.
