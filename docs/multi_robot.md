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
| 0 | learner:8102 | 172.16.0.1:4242 | 172.16.0.2 | 50055 / 50054 |
| 1 | learner:8103 | 172.16.0.1:4243 | 172.16.0.3 | 50051 / 50052 |

Learner ports are `client_port + robot_index`. Increase `num_robot` and add the
corresponding rollout processes and independent hardware endpoints for more robots.
Robot indices and their hardware mapping must remain fixed when resuming.

## DROID fork

`scripts/multi_robot/setup_droid.py` pins
[kevin-on/droid](https://github.com/kevin-on/droid) at
`5f9df37cf10153868a2f05eab43d6c32a68801bc`. This commit integrates the NUC's
local changes with per-robot RPC/Polymetis routing, camera ownership and attach
mode. No separate patch is applied.

Install the workstation dependency before client environment setup:

```bash
python scripts/multi_robot/setup_droid.py
# For local development before publishing the DROID commit:
python scripts/multi_robot/setup_droid.py --source /scr/kevinon/workspace/droid
# A separate NUC checkout can be selected when preparing deployment:
python scripts/multi_robot/setup_droid.py /path/to/new/droid
```

Publish the DROID integration branch before using the GitHub source on another
machine. The script clones new checkouts at the exact commit. At that revision,
rerunning it preserves local configuration and reports local changes; it refuses
a different revision or an existing directory inside another repository.
It does not launch hardware, install dependencies or update the active NUC
checkout at `/home/iliad/khhung/expoft`.

The fork retains the NUC's `/home/iliad/Utilities/miniconda3` + `expoft`
launcher environment, per-port logs, startup/gripper readiness retries and
cooperative waits with a 30-second ZeroRPC heartbeat. Existing 30 Hz environment
and IK settings and gripper/move tuning are retained. Controller restarts only
terminate process groups started by that `FrankaRobot` instance; launchers no
longer kill by process name, port or device. Existing controller ownership must
be resolved explicitly. NUC sudo authentication remains local DROID configuration.

For a later deployment, activate the NUC `expoft` environment and start one
server per terminal from the **new** DROID checkout, after checking active
controllers and listeners:

```bash
python -m scripts.server.run_server --port 4242 --robot-ip 172.16.0.2 \
  --robot-port 50055 --gripper-port 50054 \
  --gripper-device /dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DA6UJOT5-if00-port0

python -m scripts.server.run_server --port 4243 --robot-ip 172.16.0.3 \
  --robot-port 50051 --gripper-port 50052 \
  --gripper-device /dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DA6UJXZ9-if00-port0
```

The original CLI names `--zerorpc-port` and `--gripper-comport` are also
supported. These FTDI paths are the current mapping; recheck after hardware
changes. DROID normally starts controllers when a rollout creates its
environment. `launch_controller=false` in robot JSON instead attaches to
**both** existing arm and gripper controllers at the configured ports; an arm
listener alone is insufficient. Creating/resetting rollout environments is
hardware operation, not a connectivity check.

## SpaceMouse teleoperation without cameras

From `/scr/kevinon/workspace/expo-ft-fork` on the workstation, run one command
per terminal after `source /scr/kevinon/env.sh`:

```bash
client/.venv/bin/python -m client.teleop_spacemouse \
  --robot-config configs/robots/robot-0.json

client/.venv/bin/python -m client.teleop_spacemouse \
  --robot-config configs/robots/robot-1.json
```

The corresponding NUC DROID server must already be running. Teleop reuses
running arm **and** gripper controllers by default. If they are not running,
append `--launch-controllers` to explicitly start them through that DROID server.
A DROID listener on 4242/4243 alone does not mean those controllers are ready.
Robot 0 uses arm port 50055 because an existing shared controller occupies 50053.

Teleop reads only the server and SpaceMouse fields from the JSON; robot 1's
missing wrist camera does not block it. It starts at the current pose without
reset, camera access, recording, or task workspace bounds. Movement/twist controls
the arm, button 0 closes the gripper, button 1 opens it, and release holds.
Ctrl+C or input failure attempts a final hold command after active control.
Translation/rotation scales default to 0.5/0.1 (normalized actions, not physical
speed units). `--device-path`, `--device-number`, `--nuc-ip`, and `--server-port`
can override JSON routing; a missing requested HID path fails without fallback.

For the workstation's legacy `collect_data` camera classification, the ignored
local `client/droid/droid/misc/parameters.py` uses the following nonsecret values:

```python
nuc_ip = "172.16.0.1"
robot_ip = "172.16.0.2"
laptop_ip = "172.16.0.10"
robot_type = "panda"
hand_camera_id = "15577469"
varied_camera_1_id = "38651013"
varied_camera_2_id = "29838012"
```

These are workstation settings; NUC credentials remain local. The task's default
side/wrist observation IDs now match robot 0. `collect_data` still uses SpaceMouse
index 0 and does not consume the per-robot JSON; use this teleop CLI for explicit
robot selection. Offline checks (mocked HID/RPC):

```bash
client/.venv/bin/python -m unittest discover -s client/tests -p test_teleop_spacemouse.py
```

## Cameras and rollout clients

Both robots now have real side and wrist cameras configured:

| Robot | Side serial | Wrist serial |
| --- | --- | --- |
| 0 | 38651013 | 15577469 |
| 1 | 29838012 | 12841040 |

Robot 1's temporary wrist placeholder was replaced and `blank_camera_serials`
is empty. The pinned DROID camera reader requests **HD1080 / 15 FPS**.
When neither depth nor pointcloud output is requested, it initializes the SDK
with `DEPTH_MODE.NONE`; `depth=False` alone previously skipped only retrieval.
Depth/pointcloud users retain the SDK depth mode, and changing those requirements
invalidates camera initialization for the next mode selection. These camera
changes are included in the pinned DROID revision above. Camera discovery and
initialization are serialized across workstation processes to avoid USB probing
collisions; frame capture still runs independently for each robot.

The optional `blank_camera_serials` mechanism remains available for isolated
pipeline tests, but the current real-robot launch wrapper rejects blank cameras.
Run `scripts/multi_robot/test_camera_capture.py --help` for camera-only validation.
No NUC camera change or learner restart is needed for these workstation settings.

Each process opens only its selected real camera serials. Use disjoint real
camera lists, including any optional recording camera; a ZED is not shared across
processes. Camera settings are forwarded to DROID; the side-camera settings
accept both `varied_camera` and EXPO-FT's `static_camera` alias. Missing selected
real cameras fail before a ZED wrapper is constructed.
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

The workstation mapping confirmed on 2026-09-20 is robot 0 → `/dev/hidraw2`
and robot 1 → `/dev/hidraw0`, as configured in the JSON files. Recheck these
paths after USB changes.
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
Use `uv --no-config` for this separate test environment: the root learner
project overrides ml-dtypes/tensorstore to versions incompatible with the CPU
test requirements.

```bash
uv venv --python 3.11 /scr/kevinon/tmp/expo-ft-multi-test-venv
python scripts/multi_robot/setup_droid.py
uv --no-config pip install --python /scr/kevinon/tmp/expo-ft-multi-test-venv/bin/python \
  -r tests/cpu/requirements.txt expo_ft/agents/vla/openpi/packages/openpi-client
JAX_PLATFORMS=cpu /scr/kevinon/tmp/expo-ft-multi-test-venv/bin/python -m pytest -q tests/cpu client/droid/tests
```

DROID's own offline tests check camera/RPC routing, CLI compatibility, attach
mode, controller ownership, partial startup cleanup and preserved NUC retry and
heartbeat behavior. EXPO-FT also verifies pinned installation, repeat setup,
local-change preservation and refusal to overwrite a different revision.

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

## Online finetuning from the mirrored two-robot SFT datasets

`--num_robot=2` automatically uses the live robot1-to-robot0 mirror convention: robot0 side right/wrist left, robot1 side
left/wrist right. Both robots remain physically controlled and reset in their
own base frames. Use the intended SFT checkpoint and its normalization assets.

Offline demonstrations are ordinary HDF5 episodes supplied through
`--dataset_path`. They must already use the policy's image views and coordinate
convention. The loader does not select camera eyes, mirror offline data, or read
SFT manifests. Existing `--num_data` and offline sampling settings still apply.

To export an already-preprocessed LeRobot v2.1 dataset with embedded RGB images:

```bash
python scripts/convert_lerobot_to_hdf5.py \
  --dataset /path/to/lerobot/dataset --output /path/to/hdf5/dataset
# Then pass --dataset_path=/path/to/hdf5/dataset to the learner.
```

The exporter preserves episode/frame order and decoded image/state/action values,
without applying mirror, resize, or normalization. It verifies each written
HDF5 against the source. Outputs are `<episode_index>/traj.hdf5`; existing output
directories are refused. The learner does not need the export metadata.

Start each WS client with the two-robot launcher, for example:

```bash
EXPO_LEARNER_HOST=127.0.0.1 \
  bash scripts/multi_robot/run_workstation_rollout.sh 0
# Run the same command with 1 for the second robot.
```

The launcher reuses `robot-{0,1}-sft-eval.json`. The learner checks the selected
camera IDs through the client before constructing the hardware environment.
Robot1 observations are reflected before inference; policy actions are reflected
back before execution; returned executed actions (including physical clipping and
SpaceMouse overrides) are reflected into the common frame for replay. Reward,
termination, workspace bounds, reset, update counts and learning objectives are
unchanged. Both actor and critic consume the same canonical replay.

The round checkpoint records which live robot is mirrored. Resuming with a
different convention, or enabling mirror mode on old physical-frame replay,
is rejected. There is no separate mirror switch for this two-robot setup.
`num_robot=1` retains its existing behavior. The core round loop accepts more
than two robots without this automatic mirror convention, but additional client
configurations and launchers must be supplied; the workstation launcher here
only configures robots 0 and 1.
