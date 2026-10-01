# Two-node split learner

This opt-in path runs one JAX learner process per GPU node. ILIAD still runs
one inference process, and the robot/client protocol is unchanged. The tested
target is two DeltaAI nodes with four GH200s each, global batch 64, UTD 20,
FSDP 1, two robots, and `offline_ratio=0` (demonstrations seed replay).

The 2026-09-28 validation used
`/scr/kevinon/workspace/expo-ft-multinode`, branch `multi-node-learner`.
The main `expo-ft-fork` checkout was not the source of this test.
Companion OpenPI is unchanged at `19c1b33951bff0a80782a4d21cb552d641555d3f`.
Use the source manifest below when reproducing the tested version. That bundle
captured working-tree changes on top of its recorded base commit; archiving
only that base commit does not reproduce the tested two-node implementation.

| Mode | Learner processes | Sampling / batch layout | Launcher status |
| --- | --- | --- | --- |
| One-node learner | One process using that node's GPUs | Existing replacement sampling / flat UTD batch | Existing `scripts/split/run_role.sh` |
| Two-node learner | One process per node, four GPUs each | Global index plan / `(UTD, batch, ...)` | `scripts/split/run_role.sh` with `DELTA_NODE_COUNT=2` |

For one-node execution, leave `EXPO_PROCESS_COUNT` unset or set it to `1`.
Do not inherit `EXPO_PROCESS_COUNT=2` from a distributed learner shell when
starting ILIAD inference or a one-node learner.

## Ownership and batch semantics

- Learner rank 0 owns the ILIAD channel, replay files, round ledger, W&B, and
  snapshot export. The WAN relay targets this node only.
- Rank 0 validates a complete paired round, then broadcasts its records over
  the JAX collective connection. Both nodes insert the same records in the same
  replay order. Only rank 0 writes the batch-PKL files. This does not add
  per-transition assembly workers or another disk archive.
- `DistributedReplaySampler` uses a common seed and the checkpointed actor
  update counter to generate all 20 global minibatches' indices before gathering
  data. Each node takes 32 of each 64-row minibatch. Selection is uniform over
  eligible `(robot, replay-row)` pairs with replacement for every pool size,
  matching single-host split/colocated sampling. Different UTD minibatches may reuse rows.
  The success-only actor pool uses a separate RNG stream. Both nodes have the
  same ordered candidate pools; the sampler needs no RPC or mutable RNG state
  to resume. Existing single-host sampling remains with replacement.
- Critic inputs retain `(UTD, local_batch, ...)` on the host and use
  `PartitionSpec(None, "batch")`. Each GPU owns its slice of every minibatch;
  reshaping a globally sharded flat batch would otherwise redistribute images
  between nodes before the gradient updates. Augmentation keeps the original
  per-image RNG order.
- Both nodes finish replay insertion before reset permission. All nodes run
  the same updates and participate in Orbax save/restore. Rank 0 exports its
  local replicated policy parameters; there is no full-policy all-gather.

The original one-process behavior remains the default. Multi-host FSDP and
separate offline-ratio sampling are intentionally rejected until validated.
Given the same seed, restored update counter and ordered replay candidates,
the distributed sampler reproduces the next indices after resume. This does
not promise bitwise-equivalent floating-point updates across different GPU
counts or reproduction of an entire uninterrupted training trajectory.

## Launch contract

Initialize JAX before touching GPUs, with these per-process environment values:

```bash
export EXPO_PROCESS_COUNT=2
export EXPO_PROCESS_ID=$SLURM_PROCID  # set INSIDE each srun task: 0 or 1
export EXPO_COORDINATOR=FIRST_NODE:PORT
export EXPO_LOCAL_DEVICE_COUNT=4
```

Run `train_pi_robo.py --split_role=learner --fsdp_devices=1 --num_robot=2
--offline_ratio=0 --batch_size=64 ...` on both nodes, with the same other
arguments, source/OpenPI revisions, SFT parameters, normalization, demos and
shared output path. Demo insertion occurs on each node. Start a transport
sidecar only on rank 0; ILIAD keeps its usual sidecar. The peer node needs no
WAN ports or SSH relay. Both nodes must see the shared checkpoint directory.
Set `JAX_COMPILATION_CACHE_DIR` to a **shared** mounted directory as well.
JAX writes distributed cache entries only from rank 0; separate node-local
caches leave the other ranks recompiling on subsequent launches. See the
[JAX cache documentation](https://docs.jax.dev/en/latest/persistent_compilation_cache.html#caching-on-multiple-nodes).

For DeltaAI, use one task per node, four GPUs per task, and
`srun --kill-on-bad-exit=1`. The validated test launcher loads
`nccl-ofi-plugin/1.18.0-cuda129`, passes the site's NCCL/FI/Slingshot environment,
binds CXI devices and the OFI dependency libraries into the existing ARM SIF,
and uses `--network=single_node_vni,disable_rdzv_get`. It requires
`NCCL_NET=AWS Libfabric` and checks the logs for provider `cxi`, so TCP fallback
does not silently count as a successful fast-path test. The container's glibc,
C++ runtime, Python, JAX, NCCL and OpenSSL remain unchanged.

The existing `scripts/split/run_role.sh` defaults to one node. Two-node runs
require `DELTA_NODE_COUNT=2`, `DELTA_NODES`, `DELTA_COORDINATOR` and
`DELTA_SHARED_JAX_CACHE`; merely setting its GPU count to eight is insufficient.
The two-node launch and container binding recipe used for validation is in
`tests/gpu/multinode_validation/`. Its dated job IDs and staging paths are
explicitly guarded and must be rechecked/adapted for a fresh allocation.

Reuse the existing relay for sequential tests on the same hosts. A changed
learner-A host requires changing its reverse-forward destination once; it does
not require changing the model transport protocol. Do not replace another
experiment's live relay or start physical WS clients for these tests.

### Real-robot training deployment status

The application supports two-node split learning; the end-to-end validation
uses recorded mock clients. The production `run_role.sh` now implements the
following launch contract. One invocation starts both learner ranks; only rank 0
owns the run lock, sidecar and W&B. Configure the profile as follows:

```bash
DELTA_NODE_COUNT=2
DELTA_NODES=gh042,gh047  # recheck allocation
DELTA_COORDINATOR=gh042:29451
DELTA_SHARED_JAX_CACHE=/shared/path/to/validated-8gpu-cache
DELTA_CHECKPOINT=/shared/path/to/selected-sft-checkpoint
ILIAD_CHECKPOINT=/iliad/path/to/the-same-sft-checkpoint
```

`DELTA_SHARED_JAX_CACHE` is the directory containing the actual eight-GPU cache
entries, mounted at `/jax-cache/sync-EXPOLearner-n8`. Both learner hosts need
the staged `fabric-libs/` from the verified OFI setup below. The checkpoint
overrides avoid silently selecting an older `$STAGE/checkpoint`. The contract:

1. Stage the same EXPO/OpenPI, SFT `params/` + matching `assets/`, tokenizer and
   demonstration dataset on both learner hosts. The dataset must produce the
   same ordered replay contents. Use the same task/model config and seed.
2. Start one `srun -N2 -n2 --ntasks-per-node=1 --gpus-per-node=4` step, with the
   resource, fabric and failure-propagation options used below. Set rank-specific
   `EXPO_PROCESS_ID` inside each task. With `apptainer --cleanenv`, pass the
   variables as `APPTAINERENV_EXPO_*`, as `compute.py:execute` does; setting only
   the login shell's EXPO variables is insufficient.
3. Reuse the verified OFI bindings in `compute.py`: the site plugin and dependency
   closure, NCCL/FI/Slingshot environment, `/dev/cxi*`, container OpenSSL, and
   `NCCL_NET=AWS Libfabric`. Both containers mount the **same shared** compilation
   cache and shared output/checkpoint directory.
4. Only rank 0 starts the learner transport sidecar, owns its port/IPC mailbox,
   logs to W&B and exports policies. It and the application must share that
   mailbox mount. Rank 1 runs the application only. Both applications use the
   same `--split_session`, `--run_name`, `--output_dir`, `--split_mailbox` argument
   and training flags; rank 1 does not access the WAN mailbox.
5. Use the existing learner arguments from `scripts/split/run_role.sh`:
   `--split_role=learner --num_robot=2 --update_type=episode --delay=0
   --replan_steps=8 --fsdp_devices=1 --offline_ratio=0 --batch_size=64
   --utd_ratio=20 --num_updates=3`, plus the dataset, model/assets, task, run,
   warmup, checkpoint and stop settings. `batch_size=64` is **global**, not per
   node. Rank 0 alone needs W&B credentials, loaded from the protected file.
6. ILIAD still uses one inference process and one transport. Its source/model
   identity must match the learner; relay destination is learner rank 0 only.
   WS client/camera/mirror configuration is unchanged. Preparing this deployment
   does not start real robot clients.

For resume, both ranks use the same checkpoint directory and `--resume
--checkpoint_buffer`, with a fresh shared split session/mailbox state and the
same demo/replay ordering. Both ranks participate in Orbax save/restore; do not
resume only rank 0. See [persistence semantics](split_training.md). The sampler
uses the restored actor update counter, not the environment transition count.

## Reproduce the validated robot-free run

These are login-node commands for the **dated 2026-09-28 allocations**, not new
allocation requests. Check `squeue`, `scontrol show job`, existing steps, free
GPUs and relay endpoints first. Run DeltaAI and ILIAD roles in separate terminals
or tmux panes. Do not run `watch_allocation.py` as a training launcher: it is the
old allocation monitor and can cancel its configured previous job.

The preserved bundle on each cluster contains `compute.py`, `wan_compute.py`,
`source.tar`, `openpi.tar`, `source-manifest.json`, and a `link` symlink to the
existing protected transport configuration. All model processes bind source
from the archives; the SIF provides dependencies. The tested source SHA256 is
`5438770c6079ba5fccc717e58ec002dc1696000922655e3cc18242d8114f74bc`.

On **DeltaAI login** (`ssh -F /scr/kevinon/.ssh/config deltaai` from WS):

```bash
set -euo pipefail
cd /work/hdd/bgqe/kon/expo-ft/validation/20260928-distributed-sampler
JOB=3246733
ATTEMPT=sampler-recheck-01  # choose a fresh value; use exactly the same on ILIAD
mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "$JOB" -o '%N')")
[[ ${#NODES[@]} == 2 ]] || exit 1
COORDINATOR=${NODES[0]}:29451
module load nccl-ofi-plugin/1.18.0-cuda129
RESOURCES=(--jobid="$JOB" --overlap --exact -N2 -n2 --ntasks-per-node=1
  -c48 --gpus-per-node=4 --mem=440G --kill-on-bad-exit=1
  --network=single_node_vni,disable_rdzv_get)
# Validates source hashes; reuses node-local inputs when already staged.
srun "${RESOURCES[@]}" --time=00:10:00 python3 "$PWD/wan_compute.py" \
  --phase stage --attempt "$ATTEMPT"
srun "${RESOURCES[@]}" --time=01:00:00 python3 "$PWD/wan_compute.py" \
  --phase run --processes 2 --coordinator "$COORDINATOR" \
  --attempt "$ATTEMPT" --rounds 8 --warmup-episodes 1 \
  > "wan-$ATTEMPT-launch.log" 2>&1
```

On **Stanford sc-codex login**, reached through scdt as described in the
[operating guide](../scripts/split/README.md#1-establish-the-current-state):

```bash
set -euo pipefail
cd /iliad/u/kevinon/outputs/expo-ft/validation/20260928-distributed-sampler
JOB=17637291
ATTEMPT=sampler-recheck-01  # same fresh value as DeltaAI
RESOURCES=(--jobid="$JOB" --overlap --exact -N1 -n1 -c12 --gpus=1 --mem=120G)
srun "${RESOURCES[@]}" --time=00:10:00 python3 "$PWD/wan_compute.py" \
  --phase stage --attempt "$ATTEMPT"
srun "${RESOURCES[@]}" --time=01:00:00 python3 "$PWD/wan_compute.py" \
  --phase run --attempt "$ATTEMPT" --rounds 8 --warmup-episodes 1 \
  > "wan-$ATTEMPT-launch.log" 2>&1
```

`wan_compute.py` selects the ARM learner / x86 inference role, expands a separate
fixture manifest, creates fresh node-local IPC state, and starts the real model
and transport processes. Loopback mock robot clients run **on ILIAD**, not WS.
Original recordings are read-only. W&B is disabled for this validation.
Use a new `ATTEMPT` on every restart: the wrapper refuses an existing mailbox.
Do not run two attempts concurrently through the same transport ports.

For a fresh allocation or different source, copy the small bundle to a **new
shared artifact directory on each cluster**. In that copy, update `compute.py`'s
`JOB` and staging/input paths; update `wan_compute.py`'s ILIAD job and staging
paths and pass the new coordinator. Stage on both learner nodes and ILIAD. The
helpers' guards intentionally reject wrong allocations. Check the matching
`link` targets/TLS expiry and retarget a relay only if its compute host changed.
Reuse a healthy relay for sequential runs. Source changes require new source
archives/manifests on both clusters; staging a new path alone does not include
WS edits. The helper's `ROOT/jax-cache-global8` must resolve to shared storage
visible from both learner nodes. Do not reuse a staging-ready marker after
changing checkpoint/fixture inputs; select a fresh stage instead.

### Results, timing and stop

- Delta logs: `wan-ATTEMPT/learner-rank-{0,1}.log`, `transport-rank-0.log`, and
  `nccl.*.log`; ILIAD logs: `inference.log`, `mock-{0,1}.log`, `transport.log`.
- Each `node-result-*.json` must say `passed: true`; both
  `learner/learner-passed-rank-*.json` must confirm eight devices, update counts
  matching completed warmup rounds and the transition-count gate,
  replay verification, `distributed_sampling_verified` and
  `restored_update_sampling_exact`. Also check ILIAD's `inference-passed.json`.
- NCCL logs must show `Selected provider is cxi`; launch sets
  `NCCL_NET=AWS Libfabric` to reject an unnoticed TCP fallback.
- Keep `source-manifest.json` with each run. Copy only logs/small result files
  to WS; checkpoints and recordings remain on compute/shared storage.
- After collecting both roles' log directories, use
  `tests/gpu/multinode_validation/summarize.py --learner PATH/learner
  --inference PATH/inference --first-round 2 --cycles 5 --output timings.json`.
  The current sampler run measured 31.72s full iteration, 12.73s for three updates
  including batch preparation, and 9.33s update end to installed-policy ACK
  (five-cycle medians). These are recorded mock timings, not physical resets.
- Stop only this test's Slurm steps on both clusters, using
  `scancel --signal=TERM JOB.STEP` after identifying them with `squeue --steps`.
  The supervisors clean up their own model/mock/transport children. Keep the
  held allocations, persistent SSH relay and unrelated sessions.

## Verification scope

The validation launcher's `compute.py --phase cpu` runs the distributed/replay
checks with JAX on CPU. If the training image lacks pytest, this phase installs
`pytest==8.3.5` into the allocation's node-local `test-deps/` directory and reuses
it on later checks. Only CPU tests bind that directory and add it to PYTHONPATH;
the shared SIF and GPU training environment stay unchanged. No separate manual
installation or approval is needed for these routine test dependencies.

`tests/gpu/multinode_update.py` provides collective/control correctness and
fixed-global-batch 4/8-GPU update benchmarks. Control tests verify exact small
and multi-chunk array messages and propagation of leader exceptions.

`tests/gpu/split_wan_smoke.py --performance --playback-hz=10` runs recorded
mock clients on ILIAD and the real replay/update/snapshot pipeline. Set the
above learner environment on both Delta ranks. `--warmup-episodes=1 --rounds=3`
is a short integration check. Warmup includes the just-completed round; the
first update can follow round 1 if its retained transition count meets batch size.
Exclude the first compiled update and final drain/checkpoint when timing cycles.
Initial and first updated GPU policy parameters get exact transfer checks;
the measured cycles retain normal transport and schema checks.

Use rollout start-to-next-start timestamps on ILIAD for the full iteration;
fixed-batch update timings alone omit replay preparation, persistence and WAN.
Final validation also reloads the Orbax checkpoint and checks every saved
batch-PKL against the original recorded transition hash and success marks.
These checks do not establish physical reset duration or real policy success.

Evidence for the current allocation is kept under
`/scr/kevinon/workspace/expo-ft-validation/20260928-gh200x8/` with source hashes,
remote paths, status and the final report. Consult its status before assuming
that a launched test has passed.

The subsequent distributed-sampler validation (CPU checks, five measured 10Hz
cycles, and checkpoint-counter/next-batch reproduction on both learner hosts)
is recorded in
`/scr/kevinon/workspace/expo-ft-validation/20260928-distributed-sampler/REPORT.md`.
