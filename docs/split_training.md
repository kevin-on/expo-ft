# Separate learner and inference machines

For the prepared DeltaAI/ILIAD deployment, see the [machine launch scripts](../scripts/split/README.md).

This is an opt-in path in `train_pi_robo.py`. `--split_role=local` (the default)
keeps the existing single-process behavior. The split path supports synchronous
EXPO episode rounds, one or two robots, and `delay=0`.

The current DROID configuration enables `pi05_omit_image_keys=("right_wrist_0_rgb",)`
and requires companion OpenPI commit `19c1b33951bff0a80782a4d21cb552d641555d3f`
on both learner and inference hosts. OpenPI is a separate ignored checkout, so
merging EXPO alone does not update it. This skips the permanently masked dummy
camera in the pi0.5 prefix; the real side and wrist inputs remain present. The
full model config, including this option, participates in snapshot identity checks.

```text
WS robot clients <--> inference application <--> shared RAM <--> transport
                                                                      |
                                                        direct TLS or SSH relays
                                                                      |
                    learner application <--> shared RAM <--> transport
```

The model processes use `Channel.send/receive/send_buffer/receive_buffer`; they have no TCP, SSH,
hostname, port, or connection-count logic for the learner/inference link. The
inference application's existing WS listener is a separate connection.

## Data and update ordering

1. Learner publishes the parameters actually used by EXPO sampling: trainable
   pi0.5 parameters (including trainable vision leaves), image encoder, edit
   actor, and target critic. It does not send optimizer moments or frozen pi0.5
   base weights. This is more than just LoRA. Checkpoint EMA parameters are not
   substituted for the live actor parameters.
2. Inference verifies frozen-base and normalization fingerprints, model/task
   settings, tensor names/shapes/dtypes, and policy version before replacing the
   whole policy. It loads its fixed base once and does not initialize optimizer
   moments, a training critic, temperature, or a target actor.
3. Both robot episodes use that version throughout the round. The existing
   mirror/camera convention is preserved: robot0 side-right/wrist-left;
   robot1 side-left/wrist-right, with RGB, Cartesian state, and actual executed
   actions transformed together. Existing WS camera-view validation remains.
4. Immediately after each completed step, inference queues the **same** record
   returned by `collect_round`: pre-action observation, actual executed action
   (including clipping/human intervention), reward, mask, terminal and HIL flag.
   It copies mutable arrays before handing them to a bounded writer queue.
   Serialization and WAN communication happen off the robot worker. Transport
   payloads are never written to disk.
   At episode end the WS hands its video buffers to a background encoder before
   returning terminal info; normal MP4 encoding no longer delays the last record.
5. Learner admits only complete, correctly versioned episodes from both robots.
   With `checkpoint_buffer`, it durably writes one batch pickle per robot episode,
   then releases the received buffers and inserts records into replay in the
   existing robot order. Partial rounds never enter replay or training.
6. Existing warmup and update rules are retained: the current multi-robot loop
   waits for 10 completed rounds before an update-capable round; `num_updates=3`
   means three calls **per round**, with each call's UTD unchanged. Zero derives
   calls from collected transition count / `step_interval`. The stop count is
   also transitions across robots, and the final round can exceed it.
7. A new version is sent after updates, then installed before the next round.
   Warmup rounds reuse the installed version. Inference uses its own RNG stream;
   its barrier RNG state is included in the learner checkpoint ledger.

### Reset during learner updates

```text
Both episodes finish (WS encodes their MP4s in the background)
  -> learner receives, durably saves the round, and inserts replay records
  -> prepare_reset for the next round
       learner: updates -> checkpoint if due -> snapshot transfer
       inference/WS: reset both robots concurrently
  -> join policy installation AND both resets
  -> capture fresh observations -> next episode
```

After admitting a complete round into replay, the learner sends `prepare_reset`
for the next round **before** updating. Inference resets both robots in background
RPC workers while waiting for the normal `admit` and updated policy. The main
inference thread alone installs/samples the policy. Once the required policy is
installed and all resets finish, `start_episode` captures a fresh first observation
and collection begins. No initial frame or success check is taken during the wait;
mirror transforms and per-step streaming remain in the existing collector.

The first reset happens only after initial policy/config validation. Warmup uses
the same reset handshake even when no policy update is needed. On the final round,
the learner sends stop instead of reset permission, so there is no extra reset.
Checkpoint/resume cursors and learning settings are unchanged. `installed` still
acknowledges policy installation; it does not claim reset motion has finished.

Reset failures are checked while waiting for admission/policy. Session aborts are
also checked while waiting for resets, before any new rollout. Shutdown closes WS
connections before joining reset workers; a reset already sent to the NUC cannot
be cancelled by closing a connection. **Robots now move during the update interval.**

Deploy these changes to learner, inference and both WS clients together. The NUC
DROID server, OpenPI and WAN transport/relay implementation do not need changes.
The non-split/local training loop, collection and evaluation keep ordinary reset.
No hardware validation is implied by the isolated tests:
`python3 -B tests/test_split_reset_overlap_isolated.py` uses only the standard
library, in-memory peers and fake robots (no sockets, GPU or hardware imports).

### Background episode video encoding

Split inference requests `async_video=True` when creating training environments.
Each WS environment transfers ownership of completed `raw`/`record` frame lists
to one `EpisodeVideoWriter` thread and immediately replaces its own buffers with
empty lists. The encoder only sees those detached frames, never cameras, robot
RPCs or the next episode's mutable buffers. `get_info_for_step` returns terminal
info without waiting for normal encoding, allowing the last transition and
episode-end message to reach the learner. The encoding/MP4 settings are unchanged.

The writer retains at most one completed episode at a time. If a previous save
still runs when the next episode ends, submission explicitly waits and logs a
backlog warning rather than accumulating unbounded full-resolution video in RAM.
Normal environment close/disconnect drains the writer; forced process termination
cannot guarantee unfinished files are saved. Encoding errors remain nonfatal and
are logged, as in the synchronous path. Collection, evaluation and non-split
training retain their default synchronous video behavior.

`python3 -B tests/test_async_video_isolated.py` checks terminal response while the
encoder is blocked, detached buffer ownership, bounded backlog, close/drain and
error reporting, without importing robot/camera code or running an encoder.

## Latency goal and RAM transport

The split exists to reduce **total experiment wall time**: use the larger GPU
allocation on Delta/DeltaAI for updates, and keep inference close to the WS on
ILIAD so rollout RPC latency stays low. Compare update time saved against the
entire post-update publication/transfer/install delay, not network MB/s alone.

`python -m expo_ft.distributed.transport --config PATH` runs independently of JAX
and NumPy. Large policies use `parallel_connections` (default 64); records/control
use a separate small-message pool (default 4). These are logical sockets; the
SSH adapter below creates the actual independent WAN connections.

The bulk path preserves the earlier compute-to-compute benchmark's mechanism:

1. Serialize the live policy directly into anonymous host RAM (`memfd`). It has
   no disk pathname; no intermediate snapshot is written into the checkpoint directory.
2. Seal the buffer against mutation and pass its descriptor to the sidecar over
   a Unix socket. Both processes reference the same pages; IPC does not copy the
   whole snapshot. This works across container boundaries when the Unix socket
   directory is visible to both; it does not need a shared `/dev/shm` mount.
3. The sidecar splits the buffer into 4 MiB chunks. Persistent TLS workers take
   the next chunk from a shared queue, sending `memoryview` slices directly and
   acknowledging each chunk. Faster links keep working while slower links finish;
   a failed link returns only its current chunk to the queue. No payload fsync.
4. The receiver preallocates one exact-size RAM buffer; each stream uses
   `recv_into` on its disjoint slice. After all ranges arrive, it verifies the
   whole-buffer XXH3-128 and seals the buffer. Only then is it visible to inference.
5. Inference receives a descriptor to those same RAM pages, deserializes and
   installs the complete candidate, waits for GPU readiness, and releases the
   receiver buffer. Frozen base weights stay resident; the snapshot is still
   the trainable actor, encoder, edit actor and target critic, not optimizer state.

Snapshots use the versioned `EXPOARR1` layout: a header, 64-byte-aligned C-order
array bytes, and a JSON manifest containing identity/version, shape/dtype and
byte ranges. GPU installation views the sealed receiver RAM directly, without
ZIP/NPY materialization or a second ZIP CRC scan. Transport XXH3 verification
and parameter identity/shape/dtype checks remain enabled; the receiver does not
scan parameter values for NaN/Inf. Host views stay alive until asynchronous device
copies finish, including on failure;
CPU installation uses owned copies because JAX can retain host buffers there.
Both peers must run this format; legacy ZIP snapshots are rejected. Persistent
checkpoint formats are unchanged. Normal model/replay checkpoints still use persistent
storage; they are not snapshot transport spools.

### Receipts, bounds and failure behavior

- A transport receipt means verified receipt in **RAM**, not disk persistence or
  replay insertion. Receiver pages are retained until application release; sender
  pages are released after the remote transport acknowledges. Small dedup receipts
  remain in RAM for the active session.
- A partial transfer is invisible. Independent link workers take jobs from a
  shared queue. A detected network failure puts only that job back in the queue
  before the failing worker backs off; healthy links keep sending and can take
  over the failed range. Completed ranges are retained. Offer/commit requests
  use the same pool, with no fixed first-link dependency. Record workers retire
  completed messages independently and can reach every configured peer endpoint.
- A repaired link rejoins without a model restart. A total routing outage keeps
  pending buffers in RAM until connectivity returns or the application times
  out. This does not make failure detection instantaneous: TCP/TLS connection
  setup uses `connect_timeout` (default 5 s), capped by `socket_timeout`; an
  established stalled socket still uses `socket_timeout` (default 120 s).
- **RAM transport process/node failure ends the session.** Generation IDs reject a restarted
  sidecar rather than silently forgetting delivered transitions or replaying physical
  commands. Restart both roles/transports with fresh session IDs/mailbox directories
  from a completed learner checkpoint. This differs from the old disk-spool design.
- The record queue remains bounded at 128. Pending sender buffers default to **8 GiB
  RAM**, receiver buffers to **16 GiB RAM**, maximum single payload to **8 GiB**.
  These are upper bounds, not preallocations. The proposed 64 GiB **disk** cap does
  not apply to this RAM path; it is not silently repurposed as a 64 GiB RAM budget.
  Transport config keys: `max_outbox_bytes`, `max_pending_bytes`, `max_message_bytes`.
- `mailbox`/`--split_mailbox` now names only a directory for the local Unix socket,
  locks and tiny diagnostic/session markers. Keep it node-local, private, short
  enough for a Unix socket path, and shared between the app and its sidecar.
- With `--checkpoint_buffer`, each complete robot episode is saved once as
  `robot-N/buffers/START-END.pkl` (12-digit global steps). The ordered raw
  transitions, including `is_success`, are unchanged. Both episode files are
  flushed/fsynced and atomically published before transport release, replay
  admission, reset permission, updates or checkpoint publication. No duplicate
  `received-rounds/*.records` archive is written; existing archives are left alone.
  Without `checkpoint_buffer`, transitions are kept in memory only.
- Resume accepts legacy per-transition `STEP.pkl` files, batch files, or a
  non-overlapping mixture. It checks that global steps through the checkpoint
  are present exactly once, then restores each robot in chronological order.
  Both split and local multi-robot checkpoints occur after complete rounds and
  therefore at batch boundaries; an interior-batch checkpoint is rejected before
  file cleanup. The lower-level replay reader supports partial batch cutoffs.
  Use `--resume --checkpoint_buffer` with the matching run and a fresh session;
  abandoned split replay suffixes are moved aside, not silently trained after
  restart. Model/optimizer and split checkpoint-ledger formats are unchanged.
- Backpressure, checksum/configuration failures and peer timeout abort the session.
  Physical commands are never retried by the transport. Do not delete a live
  mailbox or stop unrelated robot processes, listeners, SSH masters or allocations.
- Payload checksums use **XXH3-128** (`xxhash>=3.0.0`) for both snapshots and small
  messages; the wire field is `xxh3_128` (32 hexadecimal characters). Both applications
  and sidecars must use this version together, with a fresh session/mailbox when
  upgrading from SHA-256 payloads. Legacy payload metadata is rejected. Message IDs
  and model/normalization fingerprints still use SHA-256. TLS/SSH security is unchanged.
  This replaces the checksum algorithm only; streaming verification/export is not
  implemented. DeltaAI hashing was benchmarked; cross-host ILIAD validation is pending.

### Timing

The learner logs `POLICY_READY` and `split/*` W&B metrics:

- `update_seconds`: the round's actual update calls, including batch preparation
  and waiting for update results.
- `export_seconds`: GPU-to-host parameter gathering and RAM serialization.
- `hash_handoff_seconds`: sender SHA-256 and local descriptor publication.
- `receive_seconds` / `verify_seconds`: receiver range transfer and full-buffer
  verification, measured on that receiver's own monotonic clock.
- `install_seconds`: deserialize, validate and install on the inference GPU,
  including waiting for the inference parameter cache to become ready.
- `through_inference_ready_seconds`: learner-observed time from the last update
  completion until the installed-version ACK returns. This includes intervening
  checkpoint/logging, export, transfer, install and control RTT. For the initial
  policy it starts before export. It does not require synchronized machine clocks.

The transport also logs `RAM_TRANSFER ... send_through_verify_seconds=...`.
Measure the real Delta -> ILIAD route before claiming the old WAN throughput is
preserved. A same-node test can validate copies/lifetimes and establish local
processing overhead, but cannot establish WAN or robot rollout latency.

## Deployment configuration

Example transport configuration on the learner host (use the corresponding peer
address and its own mailbox on the inference host):

```json
{
  "mailbox": "/local/scratch/RUN/learner-mailbox",
  "listen": ["0.0.0.0", 19001],
  "peers": [["INFERENCE_HOST_OR_FORWARD", 19002]],
  "parallel_connections": 64,
  "record_connections": 4,
  "chunk_bytes": 4194304,
  "token_file": "/private/expo-link/token",
  "tls": {
    "cert_file": "/private/expo-link/cert.pem",
    "key_file": "/private/expo-link/key.pem",
    "ca_file": "/private/expo-link/pinned-ca.pem"
  }
}
```

Provision a shared random token (32+ characters) and site-approved/pinned TLS
certificates outside the repository, with private-file permissions. TLS remains
enabled through relays. The explicit `allow_plain_loopback` option exists only
for isolated tests and is rejected for non-loopback listeners/endpoints. Both
ends must use the same `chunk_bytes` (default 4 MiB) for small-message routing.
The receiver chooses the bulk chunk layout. At most 8,192 chunks are allowed
per message to bound the missing-index header (default 8 GiB messages need only
2,048 chunks). Never put credentials into tracked JSON.

If direct compute-to-compute access is unavailable, run
`python -m expo_ft.distributed.relay --config PATH` on the authorized relay host:

```json
{
  "ssh_config": "/private/ssh/config",
  "ssh_host": "REMOTE_LOGIN",
  "connections": 64,
  "listen_host": "127.0.0.1",
  "first_port": 20000,
  "destination": ["REMOTE_COMPUTE", 19002],
  "reverse": {
    "listen_host": "127.0.0.1",
    "first_port": 21000,
    "destination": ["LOCAL_COMPUTE", 19001]
  },
  "output": "/local/scratch/RUN/relay-endpoints.json"
}
```

Each SSH connection has both `-L` and `-R`: snapshots and transitions can share
the same 64 persistent SSH/WAN connections in opposite directions. The adapter
disables SSH multiplexing for these connections, never forwards an agent, and
cleans up only its own SSH children. It emits `peers` and `reverse_peers`; put
these lists into the appropriate transport configurations. TCP application
sockets still exist separately in each direction.

For the measured DeltaAI → scdt → ILIAD route, put this before the host entries
in the **task-specific** `ssh_config`, so it applies to both the ProxyJump host
and the final compute host (a command-line `-c` on the final SSH alone does not
configure the jump process):

```sshconfig
Host *
  Ciphers aes128-gcm@openssh.com
  ControlMaster no
  ControlPath none
  ForwardAgent no
  Compression no
```

Use 64 distinct `peers` and `reverse_peers` from the relay with 64 bulk workers.
Leave `IPQoS` unchanged. This is a measured setting for this route; other sites
can select different connection counts/ciphers through configuration without
changing the learner or inference code. Do not edit global SSH settings.

The defaults expose forwarded ports only on each relay's loopback. A compute
host cannot dial another machine's `127.0.0.1`: supply an approved internal
forward to reach those listeners, or reachable bind/advertise addresses when
site policy permits. OpenSSH `GatewayPorts` policy can restrict reverse binds;
do not assume requesting a non-loopback address makes it accessible. Configure
and verify both routes before starting model applications. This adapter does
not install credentials or change sshd/firewall policy. Each SSH child has its
own supervisor: startup failures/disconnections restart only that child using
the identical local/reverse ports, with 1–30 s exponential backoff. Healthy
children keep running during recovery. Shutting down the adapter cleans up all
its own children. The endpoint JSON is published while links start; it describes
stable routes, not proof that every link is ready. `first_port` and
`reverse.first_port` also preserve endpoints across whole-adapter restarts, so
live RAM transports can reconnect. If endpoints change, restart a fresh session
with the updated configuration.

The workstation checkout is the source of truth. Commit there, package the source
with its commit ID and archive SHA-256, and distribute the **same archive** to both
roles. Extract into a new hash-named directory and bind it read-only into the
runtime. Include the separately pinned OpenPI revision/archive in provenance;
the container image alone does not identify application code. Do not patch a
running remote source directory or assume workstation edits reach remote jobs.

## Model launch

Use the same tested source revision, OpenPI revision, model config, base/SFT
weights, normalization assets, and task config on both machines. Expose exactly
one GPU to inference; the learner uses existing `fsdp_devices`/batch constraints.
Start both transport processes first. The common application arguments are:

```bash
--config=configs/model/expo_ft_pi_config.py
--config_task=configs/task/pick.py
--config.pi05_weight_loader_path=CHECKPOINT/params
--config.pi05_assets_dir=CHECKPOINT/assets
--config.pi05_asset_id=MATCHING_ASSET_ID
--num_robot=2 --replan_steps=8 --delay=0 --update_type=episode
--split_session=FRESH_RUN_ID
```

Run `python train_pi_robo.py` with those arguments on each machine, adding:

| Role | Additional arguments |
| --- | --- |
| Learner | `--split_role=learner --split_mailbox=LOCAL_LEARNER_MAILBOX --dataset_path=CANONICAL_DEMOS --num_updates=3 --batch_size=8 --utd_ratio=20 --fsdp_devices=1 --max_steps=DESIRED_COUNT --output_dir=PERSISTENT_OUTPUT --run_name=RUN --checkpoint_model --checkpoint_buffer --checkpoint_interval=2000` |
| Inference | `--split_role=inference --split_mailbox=LOCAL_INFERENCE_MAILBOX --client_host=APPROVED_BIND_ADDRESS --client_port=8102 --output_dir=VIDEO_ROOT_ON_WS --run_name=RUN` |

Demo initialization on the learner is unchanged. Supply demos already in the
chosen canonical frame. Inference needs weights/assets, not the demo dataset.
For the existing two-robot WS clients, point learner endpoints to the **inference**
host/ports (8102/8103 in this example). Video paths still refer to the workstation
that writes the videos. Creating real client environments can launch/reset
controllers: preparing GPU roles does not authorize starting robot clients.

## Validation commands

Run these only on an idle, authorized test allocation, not a live robot WS:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests/distributed -v
python tests/gpu/split_smoke.py --dataset RECORDED_DEMO \
  --params PI05_BASE_PARAMS --output NEW_TEST_OUTPUT
```

The GPU test has two independent model processes and two real socket transports,
but substitutes a recorded-observation environment. It tests 24 episodes, warmup,
two real updates, a new snapshot installation, fixed-RNG action parity with the
learner, and checkpoint save/restore. It uses batch 2, UTD 1, N=2/edit=2 to keep
validation short; it is not a throughput benchmark or a cross-cluster link test.

### Previous disk-spool validation — 2026-09-25 (superseded transport)

Implementation starts from EXPO-FT `b6ce2432a0ea493d976ad769fe1231cc0cb01a20`,
with OpenPI `5f1c132e36098dba55bc21a042dbef5d3b539fec`, JAX 0.5.3, ARM runtime
on DeltaAI GH200 allocation `3221988`, node `gh138`.

- `tests/distributed`: 14 passed (TCP and TLS transfer/reconnect, complete-round
  admission, mirrored executed/HIL actions, resume cursor, and mocked bidirectional
  32-SSH command construction).
- Existing `tests/cpu/test_mirror_online.py`: 2 passed.
- GPU integration: 24 recorded-observation episodes / 384 transitions, batch 2,
  UTD 1, N=2/edit=2, two real updates, policy versions 0 and 352. Both installed
  policies matched learner actions with the same RNG (`rtol=1e-4, atol=1e-5`).
  Checkpoint 384 restored with actor step 2. Final snapshot size: 2,017,585,738
  bytes. The final integration step exited 0.
- Existing pytest-based replay/round suite was not run: this runtime does not
  contain pytest. No runtime dependency installation was performed.
- No WS hardware, robot RPC, camera, ILIAD process, or ILIAD connection was used.
  Production cross-cluster routing and WAN throughput for this implementation
  remain unvalidated; all real socket tests ran within the DeltaAI compute node.
  No performance claim is made for batch 64, multi-GPU, or physical robot control.

Evidence is retained on shared storage under
`/work/hdd/bgqe/kon/expo-ft/split-validation-20260925/`: `gpu-final/` (model logs,
checkpoint and `test/PASSED.json`), `gpu-final.log`, `final-checks-3.log`, and the
tested `source/`. Test steps and listeners were stopped; the held allocation was
preserved.

### RAM benchmark

On an authorized idle test node, with the same actual snapshot used previously:

```bash
python tests/distributed/benchmark_transport.py --payload SNAPSHOT.npz \
  --output NEW_RESULT_DIR --connections 64 --trials 3
```

This starts two TLS sidecars and two independent application processes, loads the
source once before timing, verifies every reconstructed buffer and releases it.
It writes only logs/results, not payload files. For an authorized cross-cluster
comparison, provision the regular transports/64 relays and run the script's
`--role sender --mailbox LOCAL_MAILBOX` and `--role receiver --mailbox LOCAL_MAILBOX`
on the two test hosts with the same `--session` and `--trials`. Only the sender
reads `--payload`; the receiver's argument is unused. Do not attach benchmark
applications to a live training mailbox.

### RAM validation — 2026-09-25

DeltaAI allocation `3221988`, node `gh138`, existing ARM/JAX 0.5.3 runtime:

- Final unit run: **17 passed**, including TCP/TLS, immutable descriptor handoff,
  concurrent records/snapshot, partial-range retry, whole-buffer corruption
  rejection, reconnect, released-buffer deduplication, mirrored executed actions,
  resume cursor and SSH command construction. Source matches this RAM revision.
- Independent-process TLS benchmark: two sidecars and two applications, 4 CPU
  cores on one host, three transfers of the **same 2,017,578,406-byte snapshot**
  used by the earlier WAN benchmark. Every SHA-256 matched. Transfer payloads
  stayed in RAM; output contains only logs/results.

| Trial | Receive/reassembly | Receiver verification | Send through verified ACK |
| --- | ---: | ---: | ---: |
| 1 | 0.762 s | 0.970 s | 1.734 s |
| 2 | 0.707 s | 0.967 s | 1.674 s |
| 3 | 0.721 s | 0.971 s | 1.693 s |

Sender hashing/descriptor handoff cost another 0.959–0.963 s. The benchmark also
independently hashes the application's received mapping (about 0.99 s), outside
the transport verification above. These are **same-host TLS measurements**, not
evidence that the earlier 76–87 MB/s WAN result has been reproduced.

- Real-model integration: two model processes and two sidecars on one GH200,
  recorded RGB only, 24 episodes / 384 transitions, batch 2, UTD 1, N=2/edit=2,
  two updates, policy versions 0 and 352. Fixed-RNG actions matched the learner
  after both RAM installations (`rtol=1e-4`, `atol=1e-5`). Checkpoint 384 saved
  and restored with actor step 2 in the learner process. The driver exited 0.
- Integration's post-update `POLICY_READY` span was 4.226 s on this single host
  with plaintext loopback; its export/install hooks also perform action-parity
  sampling. It is **not** an isolated serialization/install benchmark, a WAN
  timing, a four-GPU scaling result, or a fresh-process resume test.
- The first unit attempt exposed rejection of a sealed `MAP_SHARED` read mapping
  on this kernel. Immutable readers now explicitly map `MAP_PRIVATE|PROT_READ`;
  the pages remain shared and cannot be written/COWed by that mapping. No system
  permissions, package installations or container changes were needed.
- GPU integration preceded only the final W&B `commit` flag adjustment; final
  unit tests cover that source. All validation used W&B disabled. Live W&B and
  production Delta/DeltaAI -> ILIAD routing remain untested for this revision.

Evidence: `/work/hdd/bgqe/kon/expo-ft/split-ram-validation-20260925/`, especially
`validation-unit-final.log`, `validation-benchmark-32/results/`, and
`validation-gpu-1/test/PASSED.json` plus the role logs. No WS/ILIAD workloads,
camera devices or robot RPCs were used. Test processes/steps were stopped; the
parent GPU allocations were retained. Policy identity completeness remains a
separate review item. Subsequent abort and link-recovery changes are described below.

### Abort while waiting for WS — 2026-09-25

The split collector now checks the local sidecar for a received peer abort every
100 ms while waiting for WS/RPCs, before starting a round, and after policy
sampling. A failed check closes all environment connections and discards the
incomplete round. The check is a local Unix-socket RPC (5-second response timeout),
not another WAN round trip. Listener startup and close are synchronized so a
cancelled client cannot reopen its listener. The legacy local collector does not
enable these checks.

This observes an abort **after it reaches the local sidecar**. It cannot recall
an RPC already executing on WS, and it is not a heartbeat that detects a killed
learner with no delivered abort. If sampling blocks, the check runs before its
result is handed to the robot worker.

DeltaAI `3221988.22` on `gh138`: **23 tests passed**, including real loopback
WebSocket waits at accept/create/reset/step, abort before a round and during
sampling, and idle-channel peer-abort detection over TCP and TLS. No real robot
or camera was used. Evidence:
`/work/hdd/bgqe/kon/expo-ft/split-ram-validation-20260925/validation-abort-1.log`.

### Independent link recovery — 2026-09-25

DeltaAI `3221988.24` on `gh138`: **34 tests passed**. TCP and TLS fault cases
include three permanently unavailable routes, partial-range disconnect, lost
commit ACK after application release, records progressing beside a stalled RPC,
total-outage recovery, and a repaired worker rejoining. Mocked SSH children verify
that initial/disconnection failures restart only that child at identical forward
and reverse ports; a healthy child is untouched. No real SSH fault was injected
by these unit tests.

The same 2,017,578,406-byte fixture was then transferred three times over 32 TLS
connections on `gh124`, using four CPU cores in allocation `3224015` and the
existing ARM image. Receive/reassembly took **0.780 / 0.753 / 0.743 s**, with
SHA-256 verification **0.979 / 0.983 / 0.985 s**; all hashes matched the earlier
fixture. The previous same-host run on `gh138` took 0.762 / 0.707 / 0.721 s for
receive/reassembly. These separate-node samples show similar local overhead,
not a controlled WAN-throughput comparison.

Evidence under `/work/hdd/bgqe/kon/expo-ft/split-ram-validation-20260925/`:
`validation-failover-1.log` and `validation-failover-benchmark-32-cpu/results/`.
The original one-GPU allocation was cancelled externally after the unit run;
no replacement allocation was requested. Test steps/children exited, and the
four-GPU parent allocation was retained. No model code changed for link recovery.

### Cross-cluster recorded integration — 2026-09-25/26

ILIAD H200 inference (17610534, `iliad-hgx-1`) and four-GH200 DeltaAI learning
(3224015, `gh124`) completed 24 recorded episodes / 1,186 transitions with
batch 64, UTD 20, FSDP1 and three updates per round. Six actor / 120 critic
updates, exact received-record hashes, version replacement (0 → 1026), subsequent
inference, and checkpoint save/restore passed. Originals were read-only and all
used source files were rehashed unchanged. This is WebSocket mock replay, not a
physical-robot or policy-success test.

The successful fault run killed three actual SSH children during a 2.018 GB
snapshot. Only those children restarted; the remaining 29 kept their PIDs.
Snapshot and later reverse transition traffic recovered. This exposed/fixed
two edge cases: port probes must permit TIME_WAIT reuse, and failed TLS send
tracebacks must release sliced memoryviews before a healthy worker can retry
and free the same RAM buffer. The latter regression fails on the old code;
**37 tests pass** on the fixed code.

The tested route terminates 32 SSH sessions on ILIAD compute through a scdt
ProxyJump, making reverse listeners compute-local. No agent is forwarded to
compute, and no firewall/sshd policy is changed. Treat forward and reverse
reachability separately when deploying at another site.

Steady update round: 32.887 s for three calls. Fault-injected snapshot receive:
28.876 s, verification 1.584 s, actual GPU installation 2.332 s. The observed
update-end → ready span (45.746 s) also includes test-only parameter re-export
and inference comparisons. These numbers do not establish a speedup over an
ILIAD-only deployment.

Installed parameter bytes and transformed inputs match exactly across hosts;
strict seeded **action** parity does not. Final diagnostic runs chose the same
candidates with maximum raw action difference about 0.004, while one earlier
comparison reached 0.214 without candidate diagnostics. That larger discrepancy
is not conclusively explained. Do not interpret transport correctness as
cross-platform numerical identity.

Full evidence, attempt history, persistence-wrapper repair and limitations:
`/scr/kevinon/workspace/expo-ft-split-validation/20260925-wan/REPORT.md`.
Remote results are in `split-validation/20260925-wan/` beneath the user's
ILIAD output and DeltaAI work directories. Test services/credentials were
cleaned up; parent allocations retained. No commit or push.

### Production WAN chunk scheduling validation (2026-09-26)

Baseline split implementation: `54bce2d`. This change replaces equal per-link
stripes with dynamically scheduled 4 MiB chunks and defaults to 64 links. On the
same DeltaAI → scdt → ILIAD route, both SSH layers used AES-128-GCM through the
task-specific config above; authenticated end-to-end TLS and full SHA-256 checks
remained enabled. Model, snapshot contents and RAM descriptor handoff are unchanged.

- **39 CPU tests passed**, including more chunks than links, slow-worker load
  balancing, partial-chunk retry, lost acknowledgments, TLS failures and relay
  restart isolation.
- Actual production sidecars transferred the existing 2,017,578,406-byte snapshot
  five times: receiver times **6.77–8.25 s**, median **7.46 s / 271 MB/s**.
  Sender-observed transfer through receiver verification: **8.81–9.59 s**.
- A separate two-transfer run killed three real SSH children during its second
  transfer: **7.40 s** receive, **8.74 s** through verification. Only those three
  children restarted; the other 61 retained their PIDs. All seven snapshots
  matched the original SHA-256; no receiver payload file was written.
- This validation used CPU-only 2-core/6-GiB steps on each compute node. It did
  **not rerun GPU updates or policy installation**: the ILIAD GPU was occupied
  by an existing training step. The full-model follow-up is recorded below.

The timings exclude sender hash/handoff, GPU export/install, SSH startup and the
benchmark's additional application-level digest. The diagnostic script initially
used a Python 3.7 subprocess option on the login node's Python 3.6; after fixing
that helper, fault injection was rerun separately. No production fault was hidden.
Evidence and exact source archive hash:
`/scr/kevinon/workspace/expo-ft-split-validation/20260926-production-net/REPORT.md`.

### Full GPU follow-up after the training allocation became idle (2026-09-26)

The same optimized executable source passed recorded-data end-to-end validation
on ILIAD H200 ×1 and DeltaAI GH200 ×4. Both roles exited successfully: 24 episodes,
1,186 transitions verified against original record hashes, six actor updates and
120 critic updates (batch 64 / UTD 20), checkpoint 1186 saved and restored in-process.
Updated version 1026 was installed with exact GPU parameter/input equality; the
last 160 transitions were collected using that version. Inputs stayed read-only.

Measured first/steady three-update blocks: **58.11 / 33.17 s**. For the updated
2.018 GB policy: production export **1.23 s**, sender hash/handoff **0.99 s**,
receiver transfer **6.66 s**, receiver SHA verification **2.16 s**, and actual
GPU installation **4.51 s**. The observed update-end → inference-ready span was
**35.88 s**, including test-only re-export, hashes and action diagnostics. It is
not a 35.88-second production network or installation measurement.

The strict cross-platform action comparison still differs (maximum selected
action difference 0.003626, same candidate selected), as in the baseline. This
does not invalidate exact parameter transfer, but is not numerical action parity
or a physical-robot success result. This follow-up did not deliberately drop SSH
links; the separate production fault test above covers that case. Idle TLS
connections reconnected automatically during the first update interval.

Final output copy to shared storage completed successfully; 1,307 output files
and their sizes matched node-local results (17,469,107,377 bytes). No independent
process restored the shared copy. Test steps, listeners and credentials were
cleaned; the held GPU allocations remain available. Detailed results are under
`gpu-net-2/` in the evidence paths above.

### Dummy-camera omission integration (2026-09-26)

EXPO merge `4d70165` combines the optimized WAN transport with `042b312`; its
companion OpenPI revision is `19c1b33`. Experimental critic candidate grouping
is not included. The same H200 ×1 / GH200 ×4 recorded-data test passed twice:
24 episodes, 1,186 verified transitions, six actor / 120 critic updates, exact
installed parameters and inputs, and in-process checkpoint save/restore.

Steady three-update time fell from **33.17 s to 28.02 / 28.06 s** (about 15.5%,
including replay preparation). Updated 2.018 GB policy receive took **6.24 / 6.73 s**.
GPU installation took **34.08 / 4.28 s**: the first-run delay was not reproduced,
but its cause remains unidentified. The new graph compiled on the first run.
Cross-host actions still differ slightly (maximum 0.00345, same candidate selected);
neither numerical parity nor physical task success is established.

Both outputs persisted to shared storage, test services/credentials were removed,
and parent GPU allocations were retained. Exact timing boundaries and evidence:
`/scr/kevinon/workspace/expo-ft-split-validation/20260926-omit-camera/REPORT.md`.

### Local multi-robot execution

`--split_role=local --num_robot=2` uses the same asynchronous workstation video
writer and deferred reset RPCs. After both episodes are saved/inserted into replay,
it starts both resets while the learner updates in the calling thread. Before the
next rollout it waits for JAX updates and both resets, then reads fresh observations.
The final round starts no extra reset. Local warmup remains 10 episodes per robot;
`--split_warmup_episodes` only controls the split learner. No inter-machine policy
snapshot or transport sidecar is needed in local mode.
