# Multi-robot EXPO-FT

Start with the [operating guide](../scripts/split/README.md) for WS/DeltaAI/ILIAD
commands. This page covers current robot configuration and training semantics.
The optional split path is described in [split architecture](split_training.md).

## Configuration and hardware ownership

`configs/robots/robot-0.json` and `robot-1.json` are the source of truth for NUC
endpoint, camera serial/eye, SpaceMouse path, reset joints and workspace bounds.
The removed `robot-*-sft-eval.json` files must not be used. Do not keep a separate
copy of device mappings in launch scripts or assume Linux HID numbering is stable.

| Robot | NUC ZeroRPC | Side / wrist RGB used by policy | Policy frame |
| --- | --- | --- | --- |
| 0 | 172.16.0.1:4242 | side right / wrist left | physical robot0 |
| 1 | 172.16.0.1:4243 | side left / wrist right | reflected to robot0 |

NUC server arguments select the physical arm, controller ports and gripper
serial. Manage those in the user's NUC checkout; a WS JSON does not configure
Polymetis or prove the controller is ready. Do not instantiate `RobotEnv` as a
read-only connection test: it can launch controllers and reset hardware.

`scripts/multi_robot/setup_droid.py` installs its explicit pinned fork and refuses
an existing different revision. Existing workstation and NUC deployments may
have deliberate later changes; inspect Git status/revision before any install.
Do not reset or replace them to satisfy the old install pin.

## Mirror and offline data

Both robots execute physical actions and resets in their own base frames.
For robot1, inference reflects the side and wrist images horizontally and applies
the same Cartesian state/action convention used by dataset conversion:

- position `[x, -y, z]`;
- orientation `[−roll, pitch, −yaw]` under the dataset's Euler convention;
- Cartesian action sign changes at zero-based indices `[1, 3, 5]`;
- gripper values unchanged.

The transform is applied to observations before policy inference and inverted
for executed actions. Replay records the actual clipped/human action in the
canonical frame. This assumes the calibrated mirror setup; it is not a generic
transform between arbitrary camera or base placements.

Offline HDF5 must already have the intended camera views and coordinate frame.
The learner does not mirror offline data or consult SFT source manifests.
`scripts/convert_two_robot_data_to_lerobot.py` writes LeRobot and HDF5 from the
same transformed frames. To export an existing transformed LeRobot v2.1 dataset:

```bash
python scripts/convert_lerobot_to_hdf5.py \
  --dataset /path/to/lerobot/dataset --output /path/to/new/hdf5/dataset
```

The export preserves decoded image/state/action values and frame order; it does
not mirror, normalize or resize again. Match the SFT checkpoint with its own
normalization assets. The retained dataset/SFT campaign references are indexed
in `/scr/kevinon/workspace/expo-ft-tools/README.md`; dated drivers need adaptation.

## Synchronous training and replay

`train_pi_robo.py --num_robot=N` with N>1 requires `update_type=episode`, `delay=0`.
Each round collects one episode per robot with one policy version. A finished
robot waits for the others; training starts only after the complete round.
One policy owner serializes inference requests/RNG access. This is not batched
multi-observation inference, nor continuous asynchronous RL.

Each robot has its own replay buffer so n-step targets/action backfill cannot
cross robot boundaries. Sampling is uniform over eligible rows across robot
buffers, not a mandatory 50/50 quota. With `offline_ratio=0`, offline demos seed
robot0's buffer once; they are not duplicated into every robot buffer. A positive
offline ratio uses the configured separate offline/online batch mixture.
Success-only actor sampling and the existing EXPO objectives remain unchanged.

Local multi-robot warmup is ten completed episodes per robot. Split warmup is
controlled by `--split_warmup_episodes` (default 10); the minimum-batch-size gate
also applies. With `num_updates>0`, that many update calls run per paired round,
each with the configured UTD. With zero, calls are derived from aggregate
transitions and `step_interval`, retaining the remainder. `max_steps` counts
transitions across robots and stops at a round boundary.

Both split and local multi-robot training overlap the next resets with updates
and request background MP4 saving. They wait for the new policy and both resets
before reading fresh observations. The last round starts no extra reset.
Collection/eval/single-robot execution have their own reset/video paths.

During human control, that robot skips policy inference and polls with the
existing zero command. Actual executed actions, terminal/HIL flags and rewards
are retained. RPC disconnections abort the round; physical commands are not
silently retried. Per-step scheduling accounts for observation/RPC/inference
latency in the period and does not issue catch-up bursts after overruns.

## Persistence and resume

Enable `--checkpoint_model --checkpoint_buffer` for recoverable online runs.
Each complete episode produces one batch-PKL under its robot's replay directory.
Names encode global step ranges; files are restored in chronological order for
each robot. Legacy one-transition files remain readable. No `round.records`
archive is needed in addition to those replay files.

Resume with matching model/optimizer checkpoint, task, dataset, robot ordering
and update settings. Records beyond the restored checkpoint are not treated as
completed training. Replay contents/order are checked, but exact sampling-RNG
continuation is not promised. A fresh SFT initialization is not online resume.
The standard split launcher deliberately refuses to overwrite a trained run;
see the operating guide before constructing a resume command.

## Teleoperation and collection

The commands below access real hardware and require authorization/readiness.
Use `/scr/kevinon/workspace/expo-ft-fork` and source `/scr/kevinon/env.sh` first.

```bash
client/.venv/bin/python -m client.teleop_spacemouse \
  --robot-config configs/robots/robot-0.json --measure-bounds

client/.venv/bin/python -m client.teleop_two_robots \
  --robot0-config configs/robots/robot-0.json \
  --robot1-config configs/robots/robot-1.json --keep-vertical --use-bounds
```

Teleop attaches to running arm/gripper controllers by default; `--launch-controllers`
is explicit controller initialization. Startup retains the current pose rather
than resetting automatically. Single-robot `r` + Enter resets to configured joints.
Two-robot commands: `0` mirrors pose 0→1, `1` mirrors pose 1→0, `r` resets 0,
`t` resets 1, `q` exits. Consult the CLI for current input handling.
`--keep-vertical` controls orientation; `--use-bounds` enables the configured
Cartesian bounds during two-robot teleop. Bounds are XYZ meters in each physical
base frame; reset joints are radians. Bounds do not constrain reset trajectories
or implement collision checking. Teleop does not require cameras.

Collection uses the same robot JSONs:

```bash
ROBOT_ID=0 NUM_EPISODES=50 SAVE_ROOT=/scr/kevinon/data/NEW_COLLECTION/robot0 \
  bash scripts/pick/collect_data.sh --keep-vertical
```

This records new data and can move hardware. The SpaceMouse path comes from the
JSON. Raw eye selection, image size and MP4 flags are documented in the collection
CLI/root README; do not infer those settings from an old recording report.
`align_side_cameras.py` displays side+wrist alignment; `test_camera_capture.py`
checks camera streams and must not run while collection owns them.

## Verification

See [tests/README.md](../tests/README.md). Mocked client tests exercise routing,
reset/vertical/bounds/recording behavior without devices. CPU replay/RPC tests
cover barriers, mixing, executed actions and persistence. GPU mock tests cover
real models and snapshot installation. None establishes physical safety or
policy success; hardware tests are separate and explicitly authorized.
