# Synchronous EXPO-FT with multiple robots

`train_pi_robo.py --num_robot=N` (N > 1) runs the existing EXPOLearner in rounds:
each robot resets and collects one episode using the current policy; a finished
robot waits; after every robot finishes, the learner inserts the episodes and
updates the policy; only then does the next round start. `num_robot=1` retains
the original single-robot loop. This mode requires `update_type=episode` and
`delay=0`. It does not implement continuous ARL or delayed Real-Time EXPO-FT.

One learner thread owns inference and policy RNG. Robot threads submit independent
requests to that thread. There is no multi-observation inference API. An update
cannot overlap rollout, and a failed robot or inference aborts the entire round;
the multi-robot RPC path does not silently recreate an environment or retry an
action after a disconnection.

Each robot has its own PiReplayBuffer, so action backfill and n-step lookahead
never cross robot streams. Batches sample uniformly across eligible rows in those
buffers, then use the existing offline mixture and success-only actor rules.
When `offline_ratio=0`, demos seed robot 0's buffer once; they are not duplicated
for every robot. Samples come from accumulated replay, not only the latest round.
Policy architecture, objectives, action chunks, UTD and n-step targets are unchanged.
Each configured online batch size has one reusable host array per field. Robot
samples fill consecutive slices directly, bypassing each replay's size-keyed
staging cache; shuffling copies the completed batch before storage is reused.
During human control, that robot skips policy inference and polls SpaceMouse
with the existing zero command. Inference resumes after control returns to the
policy. The handoff's executed zero action is recorded with its observation and
outcome, including termination if the episode ends at handoff.

The existing 10-episode warmup counts **completed episodes across all robots**;
updates start in a round whose start already has at least 10 completed episodes.
With `num_updates>0`, that many existing update calls run **per round**, not per
robot. With `num_updates=0`, the sum of collected transitions across the round is
divided by `step_interval`, carrying the remainder. Each call still uses
`utd_ratio`. Warmup transitions do not accumulate a catch-up update burst.
`max_steps` counts aggregate transitions and stops after a complete round, so
the final step may exceed it. Reset starts with the next round, after updating.
Terminal reward/mask are read after the executed action, and no action is issued
after that episode's terminal observation.
Control pacing uses the previous command's dispatch time: observation, RPC and
inference latency consume the period instead of adding an extra period of sleep.
When processing exceeds the period, the next command is sent immediately and
the schedule starts from that actual dispatch; there is no catch-up burst.

## Endpoints on one workstation and one NUC

| Robot | Workstation rollout process → learner | Workstation → NUC ZeroRPC | NUC → Franka | Polymetis arm / gripper |
|---|---|---|---|---|
| 0 | learner:8102 | 172.16.0.1:4242 | 172.16.0.2 | 50051 / 50052 |
| 1 | learner:8103 | 172.16.0.1:4243 | 172.16.0.3 | 50061 / 50062 |

Learner ports are `client_port + robot_index`. Increase `num_robot` and add the
corresponding rollout processes and independent hardware endpoints for more robots.
Robot indices and their hardware mapping must remain fixed when resuming.

## DROID patch

DROID is an ignored dependency checkout. The tracked patch
`patches/droid-multi-robot.patch` targets pd-perry/droid revision
`076cecd2c892e644fdc106f8ba3a79482ed6e0e8` (the EXPO-FT fork). It makes the
ZeroRPC/Polymetis ports, robot IP, gripper serial device and camera ownership
explicit. Controller restarts terminate only the process groups owned by that
FrankaRobot instance; the global `pkill` commands are removed.

Apply on both workstation and NUC, before the normal DROID environment setup:

```bash
python scripts/multi_robot/setup_droid.py
# Or select a separate pinned NUC checkout:
python scripts/multi_robot/setup_droid.py /path/to/droid
```

The script is idempotent and refuses a different base revision. It does not
launch hardware, install system drivers or edit an existing different revision.
Existing DROID Polymetis configuration/real-time permissions remain prerequisites.
NUC sudo authentication still uses DROID's `sudo_password` configuration.

Start one NUC server in each terminal, from that patched DROID directory:

```bash
python scripts/server/run_server.py --port 4242 --robot-ip 172.16.0.2 \
  --robot-port 50051 --gripper-port 50052 \
  --gripper-device /dev/serial/by-id/REPLACE_WITH_FULL_DA6UJOT5_DEVICE_NAME

python scripts/server/run_server.py --port 4243 --robot-ip 172.16.0.3 \
  --robot-port 50061 --gripper-port 50062 \
  --gripper-device /dev/serial/by-id/REPLACE_WITH_FULL_DA6UJXZ9_DEVICE_NAME
```

Use the **actual full symlink names** under `/dev/serial/by-id`; the serial alone
is not a device path. DROID starts each controller when its rollout client creates
the environment. `launch_controller=false` in robot JSON instead attaches to
controllers that are already running on the configured ports.

## Cameras and rollout clients

The example JSON files select the provided wrist and side serials. Robot 1's
second wrist serial is deliberately unset. The listed three cameras include only
one wrist; the default side+wrist policy needs a separately assigned wrist view
for robot 1 before using that example. This patch does not share a ZED across
processes or invent a second wrist view. Each process opens only its listed
camera serials. Use disjoint camera lists, including any optional recording camera.
Both robots should use the same task/prompt and observation/action conventions;
override robot-specific bounds and reset joints in their JSON if necessary.
Overrides of existing NumPy array fields are converted to that field's dtype;
camera serial lists and other settings keep their existing types.

Run each rollout in a separate terminal on the workstation, using its client
venv. Separate terminals also keep manual success/reset keyboard input independent.

```bash
python -m client.run_client --host LEARNER_HOST --port 8102 \
  --config-task-path configs/task/pick.py --robot-config configs/robots/robot-0.json

python -m client.run_client --host LEARNER_HOST --port 8103 \
  --config-task-path configs/task/pick.py --robot-config configs/robots/robot-1.json
```

Replace each example's `spacemouse_device_path` with that robot's actual HID path.
List the paths in the workstation client environment:

```bash
python -c 'import pyspacemouse as sm; [print(d.path, d.product_string, d.serial_number) for d in sm.Enumeration().find()]'
```

The wrapper resolves the model from that path's VID/PID and opens the selected
path with the matching decoder, including when the two SpaceMouse models differ.
Paths must match the enumeration output exactly; recheck after USB changes.
Without a path, `spacemouse_device_number` retains the legacy PySpaceMouse 1.x
model-local index behavior, suitable only for selecting devices of the same
first-detected model. Use the explicit paths for the multi-robot examples.
`enable_spacemouse=false` disables HIL for a rollout. On disconnect the rollout
releases its cameras.

On the learner, add these flags to the existing EXPO training command (retain
your model, dataset, task and run-name flags):

```bash
--num_robot=2 --client_host=0.0.0.0 --client_port=8102 \
--update_type=episode --delay=0 --num_updates=3
```

Videos and saved transitions use `robot-0`, `robot-1`, ... subdirectories.
Checkpoints are saved at completed round boundaries, with a `round-STEP.json`
ledger. Resume requires both `checkpoint_model` and `checkpoint_buffer` from
the original run and `--checkpoint_buffer --resume` on restart. Use the same
dataset, task, number/order of robots and update settings. The replay restore
cutoff is the restored model step; saved transitions after that checkpoint are
discarded on resume before collecting a replacement round. Exact replay sampling RNG continuation is
not promised, matching the original replay restore behavior.

## CPU verification

Use a separate Python 3.11 test environment; the production environments and
their lockfiles are unchanged. Clone OpenPI's `expo_ft` branch as described in
README first (only its lightweight openpi-client package is installed here).

```bash
uv venv --python 3.11 /tmp/expo-ft-multi-test-venv
python scripts/multi_robot/setup_droid.py
uv pip install --python /tmp/expo-ft-multi-test-venv/bin/python \
  -r tests/cpu/requirements.txt expo_ft/agents/vla/openpi/packages/openpi-client
JAX_PLATFORMS=cpu /tmp/expo-ft-multi-test-venv/bin/python -m pytest -q tests/cpu
```

Tests exercise 1/2/3-worker barriers, updated policy versions, terminal and HIL
transitions, failure cancellation, real localhost WebSocket round trips,
the real rollout request handler with fake hardware, real PiReplayBuffer
insertion/n-step sampling and BatchProcessor mixing on CPU JAX, and the training
driver's warmup/update/save/restore path with a fake policy. Regression tests use
1280-row batches over 30 updates to check fixed storage reuse (small test images),
JSON overrides of the actual pick task's array fields, and sustained human control
followed by policy handoff. Timing tests measure dispatch intervals with
observation/RPC/inference delays and per-robot pacing. Device selection tests
execute the pinned PySpaceMouse library against simulated HID devices of matching
and differing models. VLA transforms and
model weights are replaced in this harness; this is not a VLA training test.
GPU model compilation/update, ZED/HID ownership, Polymetis controller launches,
real-time scheduling of two arms on one NUC, and physical task behavior require
the target machines and hardware.
