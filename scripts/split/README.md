> Model/config contract: [docs/model_config.md](../../docs/model_config.md).
> New EXPO runs use `--initial_sft_checkpoint`; old CLI-only metadata needs explicit migration.

# DeltaAI learner / ILIAD inference: operating guide

Start here for this deployment. The workstation checkout is
`/scr/kevinon/workspace/expo-ft-fork`, branch `multi-robot`. OpenPI is a separate,
ignored checkout at `expo_ft/agents/vla/openpi`, also branch `multi-robot` from
`kevin-on/openpi`; EXPO merges/pulls do not update it automatically.
The `expo-ft-split` worktree is not an input to the prepared learner/inference.

This guide covers an existing deployment, a fresh allocation, real robot clients,
and robot-free verification. [Architecture](../../docs/split_training.md) explains
ordering and recovery. [Robot configuration](../../docs/multi_robot.md) explains
mirror, replay and hardware routing. [Test inventory](../../tests/README.md)
separates regressions from GPU and hardware checks.

**Two-node / eight-GH200 learner:** follow the
[two-node guide](../../docs/multinode_learner.md) before using this recipe.
The commands below launch a **one-node** learner. Setting `DELTA_GPUS=8` in
this profile does not enable two-node execution; that needs one JAX process
per node, distributed initialization and the fabric/container bindings.
The two-node changes currently live in the `multi-node-learner` worktree at
`/scr/kevinon/workspace/expo-ft-multinode`; do not assume the main fork checkout
or an older prepared source snapshot already contains them.

## 1. Establish the current state

On WS (lightweight commands only):

```bash
source /scr/kevinon/env.sh
cd /scr/kevinon/workspace/expo-ft-fork
git status --short --branch
git rev-parse HEAD
git -C expo_ft/agents/vla/openpi status --short --branch
git -C expo_ft/agents/vla/openpi rev-parse HEAD
tmux list-panes -a -F '#S:#I.#P #{pane_current_command} #{pane_current_path}'
ssh -F /scr/kevinon/.ssh/config -o BatchMode=yes deltaai 'hostname; squeue -u kon'
```

For Stanford login, start from WS:

```bash
ssh -t -F /scr/kevinon/.ssh/config scdt 'cd /tmp && exec bash --noprofile --norc'
# scdt is a gateway, not the Slurm controller:
ssh -t sc-codex 'cd /tmp && exec bash --noprofile --norc'
# On sc-codex:
squeue -u kevinon
```

Read `deltaai_iliad.env`. Job IDs, nodes, addresses and `/tmp` paths are live
state, not permanent constants. Before using an allocation, inspect
`scontrol show job JOB -o`, `squeue --steps -j JOB`, and its GPU processes.
Do not start model/transport processes on an occupied port/GPU. No device checks
or robot client startup are implied by a request to prepare servers.

Last verified state, **2026-09-27**, not a reservation guarantee:

| Item | Prepared value |
| --- | --- |
| EXPO application / companion OpenPI | `db268da` / `19c1b33` |
| DeltaAI learner | job `3237441`, node `gh090`, GH200 x4 |
| ILIAD inference | job `17637291`, node `iliad-hgx-1`, H200 x1 |
| DeltaAI relay host | `gh-login03`, `172.28.80.9` |
| Running WS agent / relay terminals | tmux `expo-wan-relay`, windows `agent`, `relay` |
| Historical relay deployment | `/work/hdd/bgqe/kon/expo-ft/split-validation/20260927-relay-reuse` |
| WS relay auxiliary files | `/scr/kevinon/workspace/expo-ft-split-validation/20260926-production-net/private` |
| Last mock verification | `/scr/kevinon/workspace/expo-ft-validation/20260927-fork-10hz-5/REPORT.md` |

The auxiliary directory named `expo-ft-split-validation` is **not** the split
checkout. Keep its pinned host keys and the profile used by the live agent.
The agent was originally launched from the old worktree, but established
transfers use remote relay code and its existing socket under
`/scr/kevinon/tmp/expo-split-links/`; do not restart it just to change a cwd.
Future launches below use fork.

## 2. Choose inputs and a run name

The checked-in profile records:

- mixed20 SFT step 4999 parameters and **their matching** normalization asset ID;
- canonical mixed20 HDF5 demonstrations, `offline_ratio=0` (seed replay);
- 2 robots, batch 64, UTD 20, 3 update calls/round, FSDP 1;
- 10 warmup episodes **per robot**, max 20000 aggregate transitions,
  model/replay checkpointing every 2000 transitions;
- W&B project `mtexpo`, group `split-mixed20-r2`;
- workstation videos under `expo-ft-fork/data/videos`.

These are the verified baseline, not an automatic selection of the newer SFT
campaign. To change SFT initialization, stage that checkpoint's `params/` and
`assets/` on both hosts; set the matching `ASSET_ID` and offline `DEMO`. Never
substitute normalization statistics from a different dataset.

Pick one descriptive, fresh `RUN` and use it on every machine, e.g.
`mixed20-r2-20260928-01`. It names output directories and the W&B run; it is not
a dataset or branch name. `SPLIT_SESSION` identifies one coordinated live session.
Both roles need the same session and fresh IPC directories on restart.

On WS, either edit `scripts/split/deltaai_iliad.env` or copy it to a trusted local
profile and pass its **absolute path** as the final script argument. Profiles
are sourced shell code. Never put tokens/private keys in them or run `bash -x`.

## 3. Inputs: reuse prepared staging or stage a fresh allocation

If the verified allocation is still alive, check the profile's source, image,
checkpoint and model-cache paths and reuse them. No fresh download/build is needed.
All GPU execution belongs inside Slurm; perform large copies on compute nodes.

Persistent inputs (check existence before use):

| Artifact | Location |
| --- | --- |
| GH200 ARM image | `/projects/bgqe/kon/expo-ft/access-runtime/expo-sft-learner-jax053-aarch64-20260923.sif` |
| H200 x86 image on ILIAD | `/iliad/u/kevinon/artifacts/expo-ft/access-runtime/expo-ft-sft-jax053-20260922.sif` |
| Tokenizer on DeltaAI | `/projects/bgqe/kon/expo-ft/baseline-20260921/openpi-cache/big_vision/paligemma_tokenizer.model` |
| Tokenizer on ILIAD | `/iliad/u/kevinon/artifacts/expo-ft/openpi-cache/big_vision/paligemma_tokenizer.model` |
| DeltaAI baseline checkpoint | `/work/hdd/bgqe/kon/expo-ft/sft-runs/balance0923-seed3-5k-20260923/runs/mixed-020/checkpoints/expo_pi05_droid_lora_finetune_sft_cartesian_state/mixed-020-seed3-b64-s42-5k-deltaai-20260923/4999` |
| ILIAD baseline checkpoint | `/iliad/u/kevinon/artifacts/expo-ft/sft-eval/balance0923-seed3-5k-20260923/mixed-020/4999` |
| HDF5 demos on DeltaAI | `DEMO` in the profile (dataset root containing numbered episode directories) |
| Prepared EXPO/OpenPI on DeltaAI | `/work/hdd/bgqe/kon/expo-ft/validation/20260927-fork-10hz-5/production/source` |
| Same prepared source on ILIAD | `/iliad/u/kevinon/outputs/expo-ft/validation/20260927-fork-10hz-5/production/source` |

For a **new allocation**:

1. Read Slurm's new node/job/resources. Update job, node, compute IP, stage and
   image/source paths in the profile. `scdt` cannot run `squeue`; use sc-codex.
2. Use `srun --jobid=JOB --exact -N1 -n1 -c... --mem=... --gpus=... --pty bash`
   from the relevant login node, within the held allocation's resources. Do not
   nest srun inside that shell. Inspect jobs before using `--overlap`; it does
   not mean a GPU is free. GPU counts: DeltaAI 4, ILIAD 1 for this deployment.
3. In each compute shell stage the architecture-correct inputs. Set `STAGE`,
   `SHARED_SOURCE`, `IMAGE_SOURCE`, `CHECKPOINT_SOURCE`, `TOKENIZER_SOURCE`,
   `BUNDLE`, and `ASSET_ID` from the selected profile/table, then:

```bash
# COMPUTE ONLY. Choose a fresh STAGE; this is not the repository or output root.
df -h /tmp
# Allow at least 40 GiB for image, weights, cache and test inputs.
mkdir -p "$STAGE/model-cache/big_vision" "$STAGE/code-$BUNDLE/repo"
cp "$IMAGE_SOURCE" "$STAGE/runtime.sif"
rsync -a "$CHECKPOINT_SOURCE/" "$STAGE/checkpoint/"
cp "$TOKENIZER_SOURCE" "$STAGE/model-cache/big_vision/paligemma_tokenizer.model"
rsync -a "$SHARED_SOURCE/" "$STAGE/code-$BUNDLE/repo/"
test -d "$STAGE/checkpoint/params"
test -f "$STAGE/checkpoint/assets/$ASSET_ID/norm_stats.json"
test -f "$STAGE/code-$BUNDLE/repo/expo_ft/agents/vla/openpi/src/openpi/models/pi0.py"
```

4. Set `DELTA_IMAGE`/`ILIAD_IMAGE` to the staged SIF and each `*_SOURCE` to
   `$STAGE/code-$BUNDLE/repo`. Preserve project/shared outputs across allocations;
   node-local source, checkpoints, SIF and compilation cache are disposable.
5. If a compute host changed, the old SSH route targets the old host. After
   stopping its users, retire **that route** and prepare/start a new relay/agent.
   If hosts/ports are unchanged, keep the existing route (next section).

The image provides libraries; source is bind-mounted over `/opt/expo-ft`.
Do not assume code baked into an old image is current. Do not run the historical
`access-setup/runtime.py` without checking its pinned EXPO/OpenPI versions.

### Deploying a new code version

For new code, publish/copy an explicit EXPO **and OpenPI** snapshot to both
shared hosts, then stage it. This example exports committed source; it excludes
uncommitted edits. Do not discard changes to make it pass.

```bash
# WS: small source archives only; no model/data packaging.
cd /scr/kevinon/workspace/expo-ft-fork
git status --short
git -C expo_ft/agents/vla/openpi status --short
# After choosing the committed revisions to deploy:
BUNDLE="$(git rev-parse --short=12 HEAD)-$(git -C expo_ft/agents/vla/openpi rev-parse --short=12 HEAD)"
SOURCE_PACKAGE="/scr/kevinon/tmp/expo-source-$BUNDLE"
mkdir "$SOURCE_PACKAGE"
git archive HEAD -o "$SOURCE_PACKAGE/expo.tar"
git -C expo_ft/agents/vla/openpi archive HEAD -o "$SOURCE_PACKAGE/openpi.tar"
(cd "$SOURCE_PACKAGE" && sha256sum expo.tar openpi.tar > SHA256SUMS)
```

Copy those small archives to a fresh shared directory on each cluster (SSH/scp
using the same authenticated routes as `workstation.sh`). On compute, verify
both checksums against the WS values, extract EXPO into `repo/` and OpenPI into
`repo/expo_ft/agents/vla/openpi/`, then use that path as `SHARED_SOURCE` above.
Record full commit IDs and archive hashes. Never copy client/.venv across
architectures or source models from the split worktree implicitly.

## 4. Prepare the new run; start an on-demand relay

On WS:

```bash
cd /scr/kevinon/workspace/expo-ft-fork
RUN=mixed20-r2-20260928-01  # choose a new name
bash scripts/split/workstation.sh prepare "$RUN"
# For a custom profile: append /absolute/path/to/profile.env
```

This creates private TLS/token files and copies small launch/config files to
both shared `RUN/launch` and `RUN/link` directories. No services or devices start.
It refuses existing output directories; inspect partial preparation after failure.
Transport TLS/token files must match between both endpoints. Certificates created
by this script expire after three days; inspect with
`openssl x509 -in PATH/link/cert.pem -noout -enddate`.

**Already-running relay on the same hosts/ports:** skip agent and relay launch.
Its old run-name merely identifies the SSH route/authentication socket. The new
application RUN uses its own TLS/token and mailbox through the same forwarded
ports; do not replace files under the live relay directory.

Use the relay only while an active job needs transfers; do not keep idle tunnels
running between experiments. On gh-login03, `ps -eo pid,ppid,args` can locate the
relay process; verify its children/endpoints rather than trusting historical PIDs.
Endpoint metadata is in `link/relay-endpoints.json`.

The English relay TUI focuses on transfer speed: connected/connecting counts,
current acknowledged-payload MB/s, bytes completed/total, elapsed time, recent
completed transfers with average MB/s and tunnel-count changes, and the last SSH
error. It shows Idle between transfers and marks missing/stale telemetry explicitly.

- `a`: add one tunnel in an unused/down slot (one attempt, no automatic retries).
- `d`: close one active/queued tunnel. In-flight chunks on that connection may retry.
- `q`, Ctrl+C or SIGTERM: close all SSH process groups owned by this relay,
  including ProxyJump children. Unrelated SSH masters/forwards are preserved.

`CONNECTIONS=32` is the initial count; the prepared route capacity is 64. Both
transport configurations retain all 64 endpoint slots so additions work without
restarting the models. The learner reads `link/relay-state.json` and schedules
only on active routes. Reverse/control traffic can retry through the remaining
routes. Closing every tunnel pauses transfers until a tunnel is manually added;
application deadlines still apply. There is no automatic SSH reconnection.

The learner writes tiny volatile `transfer-stats.json` snapshots twice per second
from a separate telemetry thread; payloads stay in RAM. The relay reads this file
on the shared Delta filesystem, with no extra SSH connection. Current speed uses
chunk ACKs; completed speed uses receiver payload time and excludes hash checking,
SSH establishment, model loading and GPU installation. Displayed tunnel counts
are sampled, so very brief changes between samples may not appear.

Fresh `workstation.sh prepare` wires these paths automatically. Existing prepared
runs need regenerated/matched configs (64 peer slots, learner `relay_state_file`
and `stats_file`, relay `max_connections`, `state_file` and `stats_file`); never
replace a running experiment's configs. Historical relays without telemetry show
"unavailable", not an invented speed. `run_relay.sh` preserves the TTY and writes
logs to `RUN/relay.log`. Use `--no-tui` for log-only operation.

**No suitable relay yet:** after preparation, start the following once.

WS dedicated terminal:

```bash
bash scripts/split/workstation.sh agent "$RUN"
```

DeltaAI **gh-login03**, dedicated terminal (first log in with
`ssh -F /scr/kevinon/.ssh/config deltaai` and check hostname):

```bash
RUN=mixed20-r2-20260928-01
cd "/work/hdd/bgqe/kon/expo-ft/runs/$RUN/launch"
bash run_relay.sh "$RUN"
```

A working DeltaAI multiplex SSH master is needed for the dedicated Stanford-key
agent's socket forward. Do not copy private keys to clusters. Connections use
32 initial SSH tunnels (adjustable up to 64), 4MiB chunks, listener ports 24101/24102 and forwarded ranges 24200–24263 / 24300–24363.
Only one model/transport session can own those ports at a time. Do not start a
second relay merely because a new model run has a different name.

## 5. Start learner and inference; then the robot clients

DeltaAI **login node**, separate terminal:

```bash
RUN=mixed20-r2-20260928-01
cd "/work/hdd/bgqe/kon/expo-ft/runs/$RUN/launch"
bash run_role.sh learner "$RUN"
```

Stanford **sc-codex login node**, after the scdt route in section 1:

```bash
RUN=mixed20-r2-20260928-01
cd "/iliad/u/kevinon/outputs/expo-ft/split-online/$RUN/launch"
bash run_role.sh inference "$RUN"
```

Use the roots printed by prepare if your profile overrides them. These scripts
launch Slurm steps themselves; invoke them outside an existing compute shell.
They bind staged source, SFT params+assets, caches and the new run's link files;
then start a transport sidecar and model. Inference waits for clients, without
opening cameras. `POLICY_READY version=0` confirms initial policy installation.
Learner W&B uses the protected key file named in the profile; never print it.

**Only when the user authorizes real robot execution**, on WS in two terminals:

```bash
cd /scr/kevinon/workspace/expo-ft-fork
EXPO_LEARNER_HOST=iliad-hgx-1.stanford.edu EXPO_LEARNER_BASE_PORT=8102 \
  bash scripts/multi_robot/run_workstation_rollout.sh 0
# Other terminal, same environment variables:
EXPO_LEARNER_HOST=iliad-hgx-1.stanford.edu EXPO_LEARNER_BASE_PORT=8102 \
  bash scripts/multi_robot/run_workstation_rollout.sh 1
```

Replace the host with the verified inference node. Despite its legacy name,
`EXPO_LEARNER_HOST` points to inference. The launcher enumerates cameras/HID and
can initialize controllers/reset/move robots when requested. It is not a harmless
connectivity probe. Never run it for mock verification.

## 6. Robot-free 10Hz verification

The maintained integration test is `tests/gpu/split_wan_smoke.py`. It uses real
models, updates, transports and WebSocket RPC, but starts two loopback **mock**
clients on ILIAD using recorded trajectories. No WS client or robot SDK is used.
The standalone harness also contains expensive diagnostic parameter/action
comparisons; use the recorded profiling snapshot below for comparable timing.

Latest reproducible profiling bundle, prepared from fork on 2026-09-27:

- WS driver/evidence: `/scr/kevinon/workspace/expo-ft-validation/20260927-fork-10hz-5/`.
- DeltaAI: `/work/hdd/bgqe/kon/expo-ft/validation/20260927-fork-10hz-5/`.
- ILIAD: `/iliad/u/kevinon/outputs/expo-ft/validation/20260927-fork-10hz-5/`.
- Each contains `runtime.py`, `current/source.tar`, `current/openpi.tar`, and
  `current/source-manifest.json`; runtime verifies archive and per-file hashes.
- Node-local fixture and checkpoint caches are under
  `/tmp/expo-xxh3-batch-JOB/`; original recordings are never modified.

On the **same verified allocations**, with no model/transport step using the
ports, choose the same fresh ATTEMPT in both commands. Keep the relay/agent alive only for the active test, then close them.

DeltaAI login:

```bash
ATTEMPT=repeat-10hz-01  # new each time
srun --jobid=3237441 --exact -N1 -n1 -c48 --mem=440G --gpus=4 --time=01:00:00 \
  python3 -u /work/hdd/bgqe/kon/expo-ft/validation/20260927-fork-10hz-5/runtime.py \
  --phase run --attempt "$ATTEMPT" --rounds 17 --playback-hz 10
```

Stanford sc-codex login, concurrently:

```bash
ATTEMPT=repeat-10hz-01
srun --jobid=17637291 --exact -N1 -n1 -c12 --mem=128G --gres=gpu:h200:1 --time=01:00:00 \
  python3 -u /iliad/u/kevinon/outputs/expo-ft/validation/20260927-fork-10hz-5/runtime.py \
  --phase run --attempt "$ATTEMPT" --rounds 17 --playback-hz 10
```

Both commands must exit 0. Under each bundle's `current/ATTEMPT/`, inspect
`exit.json`, `output/node-result.json`, role logs and
`output/learner/learner-passed.json` / `output/inference/inference-passed.json`.
Require 34 batch replay files verified, 2033 transitions, checkpoint restore,
matching installed versions, and the two `installed-exact-*.json` confirmations.
Ten warmup rounds, first compilation/update and final drain are excluded;
rounds 11–15 are the five measured cycles. Each has 80 steps (~8s at 10Hz).
`profile-*.jsonl` contains `round_start`, `round_rollout_done`, save/install spans
and update metrics; the local `analyze.py` documents the timing boundaries.
For collection of a different ATTEMPT, change `five-10hz` in a copy of the small
collect/probe scripts; never overwrite the previous evidence.

This reruns the **frozen tested version**, not future checkout edits. To validate
new code, `prepare.py` in the WS evidence directory packages fork and applies
only timing/exact-parameter instrumentation. Run a copy under a fresh directory,
inspect its source diff/manifest, and copy its new archives/runtime to fresh
remote bundle directories. Update the driver endpoints/output roots and attempt;
never overwrite a running or previous bundle.

For **new allocations**, do not run this frozen runtime unchanged: its ARM image
path and existing fixture staging assume the observed jobs. Adapt a copy's image
path, input staging and relay-link mount, stage the unchanged fixture from the
previous shared bundle listed below, and verify its manifest. The underlying maintained harness is allocation-independent:

```bash
# Inside the appropriately bound compute container, one role per host:
python -u tests/gpu/split_wan_smoke.py --role node --node-role learner \
  --fixture /fixture --output /output --params /checkpoint/params \
  --assets /checkpoint/assets --mailbox /mailbox \
  --transport-config /link/transport-learner.json \
  --rounds 17 --playback-hz 10 --session NEW_SHARED_SESSION
# On ILIAD: node-role inference, transport-inference.json, separate output/mailbox.
```

Persistent original fixture directories (both have `manifest.json`, `reference.pkl`
and per-robot episode files):

- DeltaAI: `/work/hdd/bgqe/kon/expo-ft/split-validation/20260927-xxh3-batch-e2e/fixture`.
- ILIAD: `/iliad/u/kevinon/outputs/expo-ft/split-validation/20260927-xxh3-batch-e2e/fixture`.

On each compute host, copy its fixture to the **new** wrapper bundle's `fixture/`
with `rsync -a`; compare the manifest checksum on both hosts. The wrapper stages
it locally and creates a separate expanded manifest, leaving original files intact.
Do not manufacture `INPUTS_READY.json` to skip staging; it records verified inputs.

The fixture manifest must contain 17 paired episodes; the latest wrapper expands
its recorded 12-round fixture to 17 by repeating the final 80-step episode. Bind
that expanded fixture read-only on both hosts. Original source records were from
`/iliad/u/kevinon/outputs/expo-ft/online/online-mixed20-r2-b64-utd20-reset-overlap-20260925-203855/output/online-mixed20-r2-b64-utd20-reset-overlap-20260925-203855/`.
Check available shared fixture directories before an allocation expires; do not
assume node-local `/tmp` is persistent.

For a two-node/eight-GH200 learner, configure the optional profile fields in
[the two-node guide](../../docs/multinode_learner.md#real-robot-training-deployment-status).
The same `run_role.sh learner RUN` launches both ranks; inference stays single-node.

## 7. Inspect, stop, restart

### Interactive rollout control on ILIAD

To control when each robot starts, set these in the deployed run profile:

```bash
ROLLOUT_DASHBOARD=true
ROLLOUT_MODE=manual
```

Run the usual inference role in an interactive terminal; use `ssh -tt` on
**each SSH hop**. The launcher supplies `srun --pty`. The two WS robot clients
still run separately with their existing commands. This feature only changes
the split inference process and works with either one or two learner nodes.

- `0` / `1`: start the corresponding READY robot.
- Space: start all READY robots.
- `m`: switch auto/manual at runtime. Auto immediately releases waiting READY
  robots; switching to manual leaves running episodes alone.
- `q`: stop inference through the existing abort path; this does not request
  a checkpoint. Request a checkpoint separately and wait for its saved message.

The dashboard shows robot status, current step, completed episodes, success
count/rate, latest result, elapsed time and human-controlled steps. Episode
totals cover the current inference session. Detailed logs go to `inference.log`.
In manual mode, new-policy installation and both resets finish before READY;
the first fresh observation is read only after that robot is started. Reset
remains automatic. Both episodes must finish before the next learner update.
Start keys outside READY are ignored rather than queued for future rounds.
Existing peer timeouts still apply while waiting for manual input.

Without `ROLLOUT_DASHBOARD=true`, the existing automatic rollout stays unchanged.

For a manual checkpoint, run this in a separate terminal on the learner's
login node (from the deployed repo):

```bash
bash scripts/request_checkpoint.sh "$DELTA_RUN_ROOT/$RUN/checkpoints"
```

Pass the existing run's **checkpoint directory**, not a numbered checkpoint.
Each Enter creates `save.request`. Local multi-robot and split learners check
it after the current round's updates, save with the existing round ledger,
wait for completion, remove the request and log `Manual checkpoint saved ...`.
Requests coalesce while one is pending, including during saving; an interval
save due at the same boundary is not duplicated. Manual requests work even
without `--checkpoint_model`; resumable replay still requires
`--checkpoint_buffer` throughout the run. With two learner nodes, rank 0
checks the file and both ranks participate in saving. Ctrl+C here exits only
the request script. This requires the updated learner code; it does not add
manual saving to an already-running older process.

- Per-run shared logs: `learner.log`, `inference.log`, `transport-ROLE.log`.
  Checkpoints/replay: `DELTA_RUN_ROOT/RUN/checkpoints`.
  Videos are written on WS to `WS_VIDEO_ROOT`, not to ILIAD's filesystem.
- For the balance0927 seed42 datasets, use W&B group
  `pick_cube_balance_0927_seed42` across dataset sizes, wrist options and GPU
  counts. Put those individual experiment differences in the run name/config,
  rather than creating a separate group for each launch.
- `robot-N/intervention_step_rate` is the fraction of human-controlled steps in
  that robot's latest episode; `training/intervention_step_rate` pools both
  episodes in the latest round (weighted by their step counts).
  `robot-N/intervention_episode_rate` and `training/intervention_episode_rate`
  are cumulative fractions of completed episodes with at least one human step.
  Warmup episodes count; demonstrations do not. Counts survive checkpoint resume;
  checkpoints predating intervention totals start metric counting anew.
  The old `intervention_rate` keys remain aliases of the step rates.
- `split/update_seconds` includes replay batch preparation and GPU completion.
  `through_inference_ready_seconds` spans update end through snapshot export,
  hash/handoff, receive/verify, GPU installation and installed ACK. It excludes
  rollout and does not assert that physical reset has finished.
- Historical one-node five-cycle median (2026-09-27): 47.68s total, 8.07s rollout, 26.27s three updates,
  9.52s update-end→ACK; receive 7.68s, GPU installation 0.21s. Mock execution is
  not proof of physical reset timing or task success. See the two-node guide
  for the separate 2026-09-28 eight-GPU results and their measurement scope.
- Stop robot clients first if running, then stop only the two role commands/test
  steps. Their handlers stop their own model/mock/transport children. Keep the
  relay only while the active experiment requires transfers. Close its TUI with
  `q` and stop its dedicated agent forward when finished; do not leave idle tunnels.
- To stop a specific Slurm step: `scancel --signal=TERM JOB.STEP` after checking
  its identity. `scancel JOB` returns the entire held allocation; do that only
  when requested. Verify remaining steps after tests.
- A fresh run is not checkpoint resume. `run_role.sh` deliberately refuses a
  trained checkpoint directory. Actual resume requires `train_pi_robo.py --resume
  --checkpoint_buffer` with the matching run/data/robot ordering and fresh shared
  split session; consult [checkpoint semantics](../../docs/split_training.md).
- If scdt/SSH authentication fails, report it. Do not create alternative relay
  hosts, change firewall rules or forward extra keys as an implicit workaround.
