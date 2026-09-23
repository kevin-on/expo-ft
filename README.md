# EXPO-FT and Real-Time EXPO-FT

Code for the papers *"EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models"* and *"Reinforcement Learning for Real-Time Vision-Language-Action Policies"*.

EXPO-FT: **[Project Website](https://pd-perry.github.io/expo-ft)** | **[arXiv](https://arxiv.org/abs/2605.25477)**


Real-Time EXPO-FT: **[Project Website](https://pd-perry.github.io/real-time-expo-ft)** | **[arXiv](https://arxiv.org/abs/2609.18207)**

EXPO-FT is the original framework for sample-efficient RL finetuning for VLAs. Real-Time EXPO-FT finetunes real-time VLA policies for high-frequency control.

To determine which is best for your task, use the following criteria:
- If your forward pass time fits inside the control step or your environment is static, use [EXPO-FT](#running-expo-ft).
- If your forward pass does not fit inside a control step and your task needs the policy to be reactive to changes in the environment, use [Real-Time EXPO-FT](#running-real-time-expo-ft).


## Setup

The repo has **two independent Python environments**:

- **Server (learner)** — `.venv/` at the repo root, managed by `pyproject.toml` + `uv.lock`. Holds the modern jax / openpi / lerobot stack used for RL training.
- **Client (actor)** — `client/.venv/`, managed by `client/pyproject.toml` + `client/uv.lock`. Holds DROID's older numpy / mujoco / opencv pins for the real-robot SDK.

Both require **Python 3.11+** and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone the forks

EXPO-FT and Real-Time EXPO-FT depend on two GitHub forks - OpenPI (used by the server) and DROID (used by the client). Clone **both before running `uv sync`**. uv installs them as editable local checkouts (see the `[tool.uv.sources]` blocks in `pyproject.toml` and `client/pyproject.toml`), so `uv sync` fails if they aren't present yet. OpenPI lives under `expo_ft/agents/vla/openpi` (used by both envs); DROID lives under `client/droid` (only the client venv needs it).


**EXPO-FT**: [modified OpenPI](https://github.com/pd-perry/openpi/tree/expo_ft) and our pinned [DROID fork](https://github.com/kevin-on/droid), including the NUC setup and multi-robot integration.

```bash
# From the repo root.
git clone -b expo_ft https://github.com/pd-perry/openpi.git expo_ft/agents/vla/openpi
python scripts/multi_robot/setup_droid.py
```


**Real-Time EXPO-FT**: [modified OpenPI](https://github.com/pd-perry/openpi/tree/real-time-expo-ft) and [DROID fork](https://github.com/pd-perry/droid/tree/real-time-expo-ft), both on their `real-time-expo-ft` branches. 

```bash
# From the repo root.
git clone -b real-time-expo-ft https://github.com/pd-perry/openpi.git expo_ft/agents/vla/openpi
git clone -b real-time-expo-ft https://github.com/pd-perry/droid.git client/droid
```



### Server (Learner)

Installs all server dependencies — including the local `expo_ft/agents/vla/openpi` checkout (editable) — via uv:

```bash
# From the repo root.
uv sync
```

### Client (Actor)

Installs all client dependencies — including the local `client/droid` checkout (editable) — via uv. The client also installs `openpi-client` from the `expo_ft/agents/vla/openpi` checkout, so that fork must be cloned too (see [Clone the forks](#clone-the-forks)).

System prerequisites (install **before** `uv sync`):

- **ZED SDK** (only if you use ZED cameras). Install from [stereolabs.com](https://www.stereolabs.com/developers/release/) — provides the system libraries that `pyzed` loads at runtime.
- **Spacemouse HID access** — see the [PySpaceMouse troubleshooting guide](https://github.com/JakubAndrysek/PySpaceMouse/blob/master/troubleshooting.md) for platform-specific setup. On Linux, you need a udev rule.

Install:

```bash
# From the repo root.
# 1. Install client dependencies into ./client/.venv.
cd client && uv sync && cd ..

# 2. (Optional) Install pyzed if you use a ZED camera. Must be a separate step
#    because the pyzed wheel's numpy>=2.0 metadata over-constrains a binary
#    that actually works against numpy 1.x, so we bypass uv's resolver.
bash client/install_pyzed.sh
```

## Overview and Code Structure

The system uses a **server-client architecture**: the **server (learner)** runs RL training with the VLA policy, while the **client (actor)** runs the DROID real-robot rollout environment and communicates over WebSocket.

```
train_pi_robo.py                # Server: synchronous RL finetuning loop
train_pi_robo_async.py          # Server: asynchronous RL finetuning loop (sampler + updater on separate GPUs)
train_offline_rtc.py            # Server: RTC-SFT base policy (action-prefix conditioning) for Real-Time EXPO-FT
eval_droid_policy.py            # Server: standalone policy evaluation

client/
  run_client.py                 # Client: environment rollout server (WebSocket)
  collect_data.py               # Client: demonstration data collection (spacemouse)
  envs/                         # Environment wrappers (DROID real robot)
  real_utils/                   # Success detectors, spacemouse, visualization

configs/
  task/                         # Task configs (pick, dynamic_pick, light2)
  model/                        # Algorithm configs
    expo_ft_pi_config.py        #   EXPOLearner
    dagger_pi_config.py         #   BCLearner
    realtime_expo_ft_pi_config.py  # RealTimeEXPOFTLearner (Real-Time EXPO-FT: real-time chunking + filtered backup)
    rtc_pi_config.py            #   RTCLearner (prefix-conditioned BC / RTC-SFT base policy)

expo_ft/
  agents/
    alg/                        # RL algorithms (Real-Time EXPO-FT, EXPO-FT, RTC-SFT, BC, base agent)
    vla/                        # VLA wrappers (pi0.5 integration)
  data/                         # Replay buffer, dataset loading, batch processor
  env/                          # Server-side env utilities (WebSocket client, dataset loading)
  networks/                     # Neural network components (encoders, critics, MLP)
  distributions/                # Action distributions (tanh normal)
  utils/                        # Logging, training utilities, augmentation, control-loop helpers (loop_utils.py)

scripts/                        # Shell scripts for launching experiments
  convert_droid_data_to_lerobot.py
  dynamic_pick/                 # Real-Time EXPO-FT example script set (dynamic pick task)
  pick/                         # EXPO-FT example script set (pick task)
```

## Use your own algorithm & VLA

You can plug in your own online fine-tuning algorithm by implementing it in `expo_ft/agents/alg`, following the learner API used by the existing agents such as `EXPOLearner`, `RealTimeEXPOFTLearner`, `RTCLearner` and `BCLearner`. Custom VLA backends can similarly be added in `expo_ft/agents/vla`, following the wrapper API used by the pi0.5 integration. After adding a new algorithm or VLA, expose it through the corresponding config in `configs/model/` so the training scripts can instantiate it.

## Running Experiments with DROID + pi0.5

For synchronous collection from multiple robots with one shared policy, see
[Multi-robot EXPO-FT](docs/multi_robot.md).

### OpenPI Setup

We use a [modified fork of OpenPI](https://github.com/pd-perry/openpi/tree/real-time-expo-ft) (`real-time-expo-ft` branch) with support for frozen encoder training (for efficient action sampling) and Cartesian action control for DROID, plus the prefix-inpainted sampling and per-token adaRMS conditioning Real-Time EXPO-FT's `--delay` relies on. It builds on the [EXPO-FT fork](https://github.com/pd-perry/openpi/tree/expo_ft). Cloned into `./expo_ft/agents/vla/openpi` and installed editable during the [server setup](#server-learner) step (see [Clone the forks](#clone-the-forks)). The same checkout provides the SFT pretraining scripts wrapped below.

### DROID Setup

Synchronous EXPO-FT uses the pinned [kevin-on/droid fork](https://github.com/kevin-on/droid); see [its setup and deployment instructions](docs/multi_robot.md#droid-fork).

Real-Time EXPO-FT uses a [fork of DROID](https://github.com/pd-perry/droid/tree/real-time-expo-ft) (`real-time-expo-ft` branch) for real-robot control, which adds the background camera-reading path needed to run the control loop at 30 Hz. Cloned into `./client/droid` and installed editable during the [client setup](#client-actor) step. For software, hardware setup and calibration, see the [DROID documentation](https://droid-dataset.github.io/droid/).

Configure the hardware-specific values before running the client.

**Client / laptop**

- `client/droid/droid/misc/parameters.py`
  - `nuc_ip`
- `configs/task/real_base.py` and any per-task overrides, e.g. `configs/task/dynamic_pick.py`
  - `side_camera_id`
  - `wrist_camera_id`

**NUC**

- NUC DROID install: `droid/misc/parameters.py`
  - `sudo_password`
  - `robot_type`
  - `robot_serial_number`
- NUC DROID/Polymetis hardware config
  - `robot_ip`

### Training Setup

1. **Environment class** -- Create an environment class for your task in `client/envs/droid_env.py`. We include our pick (`PickBlocksEnv`) and light-plug (`Light2Env`) environments as references; modify one to match your task's observation space, action space, and reset behavior. A task that only changes reset behaviour or success thresholds can reuse an existing class -- `dynamic_pick` runs on `PickBlocksEnv` with different constructor arguments.
2. **Task config** -- Create a task config in `configs/task/` to specify task-specific parameters (bounds, reset joints, language instruction, etc.). See `configs/task/pick.py` for a standalone example and `configs/task/dynamic_pick.py` for one that inherits from another task config and overrides only what differs.
3. **Success detector** -- Define a success detector for your task in `client/real_utils/detector.py` and register it in your environment class's `detect()` method.

### Running the Experiment

Two complete script sets ship with the repo: `scripts/dynamic_pick/` runs [Real-Time EXPO-FT](#running-real-time-expo-ft), and `scripts/pick/` runs [EXPO-FT](#running-expo-ft). The shared steps below are written with `scripts/dynamic_pick/` and `scripts/pick/` (or your own script directory) for Real-Time EXPO-FT and EXPO-FT.

The data steps below are shared by both pipelines; after them, select one pipeline end to end based on which method you want to run. 

> **Filesystem note:** The example scripts assume the client and server can see the same repo-relative paths. If they run on different filesystems, collect data on the client/robot machine, then copy or sync the collected `data/...` directory to the server/GPU machine before running conversion, norm stats, SFT, RL training, or evaluation. The `dataset_path`, OpenPI assets, SFT checkpoints, and RL checkpoints in the server scripts are server-local paths. The client only needs the robot environment code, task config, and network access to the learner; keep any task config or environment changes synced on both machines.

#### Data collection

Collect demonstration data using a spacemouse. The NUC must be running the DROID server before starting collection:

```bash
# On the NUC, from the DROID install root 
python scripts/server/run_server.py
```

```bash
# On the client / robot machine.
bash scripts/dynamic_pick/collect_data.sh
```

Parameters to update in `collect_data.sh`:

- `--save_root` -- output directory for collected episodes
- `--num_episodes` -- number of demonstrations to collect
- `--task_config` -- task config for the robot environment
- `--robot_config` -- optional JSON selecting the NUC port, camera pair, and SpaceMouse
- `--save_right_images` -- defaults to true; add side/wrist right views to HDF5 and MP4. Use `--nosave_right_images` for the previous left-only collection settings.

For one-at-a-time collection with the configured two-robot setup, run from the repo root
after the selected robot is available and its NUC server is ready:

```bash
ROBOT_ID=0 bash scripts/pick/collect_data.sh
# Or, for the other robot:
ROBOT_ID=1 bash scripts/pick/collect_data.sh
```

The pick script defaults to 15 successful episodes and saves each robot separately under
`data/pick_cube_balance/robot0/success/<episode>/` or `robot1/success/<episode>/`.
Override with `NUM_EPISODES` and `SAVE_ROOT`; do not share a save root between concurrent collectors.
Point subsequent conversion/training scripts at the selected robot's `success` directory.
Check the camera/SpaceMouse mapping before collection; this command opens hardware and resets the robot.

Each successful episode contains `traj.hdf5` and `recordings/MP4/<image_key>.mp4`.
With stereo collection enabled, `saved_observation` in HDF5 and the MP4 directory contain
`exterior_image_1_left`, `exterior_image_2_left` (the existing duplicate side-left view),
`exterior_image_1_right`, `wrist_image_left`, and `wrist_image_right`.
Images are resized to the task's `image_size` (default 320x180 pixels, width x height);
MP4 also defaults to 320x180 without automatic macroblock resizing. These are not 1080p originals.
Failed episodes are discarded. Stereo settings apply only to collection; adding right views
to files does not add them to the policy's training inputs.

#### Data conversion

Convert collected data to LeRobot format for pi0.5 finetuning:

```bash
# On the server / GPU machine.
bash scripts/dynamic_pick/convert_data.sh
```

Parameters to update in `convert_data.sh`:

- `MAX_EPISODES` -- max number of collected episodes to convert
- `TASK_CONFIG` -- task config used to interpret the raw DROID data
- `DATA_DIR` -- source directory containing successful demonstrations
- `REPO_NAME` -- LeRobot dataset repo/id written into the converted dataset

For a new task, copy one of the wired script directories (`scripts/pick/` or `scripts/dynamic_pick/`) and update the dataset paths, task config, checkpoint paths, and OpenPI asset IDs.

#### Normalization statistics

First time only for a new task. This is a thin wrapper around OpenPI's `expo_ft/agents/vla/openpi/scripts/compute_norm_stats.py`, so it requires the OpenPI checkout from [Clone the forks](#clone-the-forks). Run it from the repo root with the server `.venv` active.

```bash
# On the server / GPU machine.
bash scripts/dynamic_pick/calculate_norm.sh
```

Parameters to update in `calculate_norm.sh`:

- `REPO_ID` -- LeRobot dataset id from the conversion step

> **Fixed-state tasks:** update the state and action standard deviations in
> the OpenPI config after computing normalization stats. Use the q01/q99 values
> and set the standard deviation to `1`.

Both pipelines consume the same dataset and the same normalization assets from here on:

| | [Pipeline: Real-Time EXPO-FT](#running-real-time-expo-ft) | [Pipeline: EXPO-FT](#running-expo-ft) |
| --- | --- | --- |
| Base policy (SFT) | `offline_train.sh` -- `train_offline_rtc.py`, action-prefix conditioned | `finetune_droid.sh` -- OpenPI `train.py` |
| Online learner | `RealTimeEXPOFTLearner` | `EXPOLearner` |
| Inference | overlapped with chunk execution (`--delay`) | blocks at each replan boundary |
| Example scripts | `scripts/dynamic_pick/` | `scripts/pick/` |


## Running EXPO-FT

The standard EXPO-FT ([arXiv 2605.25477](https://arxiv.org/abs/2605.25477)) pipeline does VLA supervised finetuning, then online RL with `EXPOLearner`. The policy plans a chunk and the robot waits for each new chunk to be denoised at the replan boundary. Complete example: `scripts/pick/`.

### 1. Finetune pi0.5 (SFT)

Pretrain the policy with supervised finetuning on the collected demonstrations. The script is a thin wrapper around OpenPI's `expo_ft/agents/vla/openpi/scripts/train.py`, so it requires the OpenPI checkout from [Clone the forks](#clone-the-forks). Run it from the repo root with the server `.venv` active.

```bash
# On the server / GPU machine.
bash scripts/pick/finetune_droid.sh
```

Parameters to update in `finetune_droid.sh`:

- `DATA_ID` / `REPO_ID` -- dataset id used for the converted LeRobot data
- `ASSETS_DIR` / `ASSET_ID` -- OpenPI normalization assets from the stats step

The provided pick script runs about 4000 steps, which was sufficient for all tasks we tested.

### 2. EXPO-FT finetuning

After pretraining, finetune the policy with `EXPOLearner`. The server and client communicate over WebSocket: the client (rollout server) runs the environment, the server (learner) runs the RL training loop.

**Start the DROID server on the NUC**:

```bash
# On the NUC, from the DROID install root (for example client/droid
# in this checkout, or the NUC's standalone DROID checkout).
python scripts/server/run_server.py
```

**Start the client** (on the robot machine):

```bash
# On the client / robot machine.
bash scripts/pick/run_policy.sh
```

Parameters to update in `run_policy.sh`:

- `SERVER_HOST` -- hostname or IP of the GPU training machine (required; the script exits if unset). Set it in the script or pass it inline: `SERVER_HOST=my-gpu-box bash scripts/dynamic_pick/run_policy.sh`. The client dials the learner directly, no SSH tunnel needed
- `--port` -- learner port; keep aligned with the learner's `client_port`
- `--config_task_path` -- task config for the rollout environment

The learner listens on `client_host`/`client_port` (`0.0.0.0:8102` in the pick scripts) and waits for the client to connect, so either side can be started first. The robot machine only needs outbound access to that port; if it cannot reach the training machine directly, forward the port with `ssh -L 8102:localhost:8102 <training-machine>` on the robot machine and dial `localhost`.

**Then start the server** (on the GPU training machine):

```bash
# On the server / GPU machine.
bash scripts/pick/run_server.sh        # synchronous
bash scripts/pick/run_server_async.sh  # asynchronous
```

<!-- What differs: `--config=configs/model/expo_ft_pi_config.py` (`EXPOLearner`), a `--config.pi05_weight_loader_path` pointing at the SFT checkpoint from step 1, and no `--delay` -- inference blocks at each replan boundary.  -->

> **Async training:** Requires ≥ 2 GPUs (1 sampler + ≥ 1 updater). Use it when one episode takes a long time; otherwise synchronous training can yield better results.

Key parameters to configure in `run_server.sh` and `run_server_async.sh`:

- `dataset_path` -- path to the collected demonstration data
- `num_data` -- max offline demo episodes to seed into the replay buffer (0 = all)
- `config` -- config set to `configs/model/expo_ft_pi_config.py` (`EXPOLearner`)
- `--config.pi05_weight_loader_path` -- the SFT checkpoint from step 1
- `update_type` / `num_updates` -- for synchronous training, recommend: use episode updates with `env_steps / num_updates` close to 20-30
- `step_interval` -- alternative to a fixed `num_updates`: one gradient update per this many collected transitions
- `edit_scale` -- edit scale
- `client_host` / `client_port` -- interface and port the learner listens on; `0.0.0.0` accepts the client from any machine


> **Client recovery:** If the client hits an error or the robot gets stuck during online training, you can stop and restart only the client. The server waits for the policy/environment connection to recover, then continues training once the client is restarted.


### 3. Evaluation

Evaluate the trained policy. Start the DROID server and the client rollout server the same way as in [EXPO-FT finetuning](#2-expo-ft-finetuning), then launch evaluation from the server/GPU machine. All model parameters should match the training configuration.

```bash
# On the server / GPU machine.
bash scripts/pick/eval_policy.sh
```

Parameters should match the corresponding `run_server.sh` or `run_server_async.sh` training settings.

#### Evaluate the mixed two-robot SFT checkpoints

Use the existing `config.pi05_weight_loader_path`, `config.pi05_assets_dir`,
`config.pi05_asset_id`, `only_base_actions`, and `checkpoint_step=0` settings.
The launcher sets `N=1` and disables action editing. It runs no model updates;
existing EXPO initialization still needs one HDF5 episode for input shapes.

Run one robot at a time, with the same robot ID on both hosts:

```bash
# Allocated GPU/container shell, in the prepared learner environment.
export SFT_CHECKPOINT=/path/to/completed/run/3889
export SFT_ASSET_ID=expo_ft/pick_mixed_100
export EXPO_DATASET=/path/to/demo/episode-parent
export EXPO_EVAL_OUTPUT_DIR=/path/to/eval/mixed100-step3889-robot0
export EXPO_CLIENT_VIDEO_DIR=/scr/kevinon/data/eval/mixed100-step3889-robot0
bash scripts/pick/eval_sft_policy.sh 0

# Workstation, when ready to move the selected robot:
export EXPO_EVAL_HOST=<reachable-compute-host-or-tunnel-endpoint>
bash scripts/pick/run_sft_eval_client.sh 0
```

`SFT_CHECKPOINT` is a completed step directory containing `params/` and
`assets/$SFT_ASSET_ID/norm_stats.json`; these paths must be visible in the GPU
container. The scripts do not allocate GPUs, start a container, or set up tunnels.
Default ports are 8202/8203; override `EXPO_EVAL_BASE_PORT` on both hosts.
`EXPO_EVAL_EPISODES` defaults to 10, `EXPO_REPLAN_STEPS` to 8, and
`EXPO_EVAL_SEED` to 42. `EXPO_EVAL_PYTHON` defaults to the active `python`.

Eval JSONs select robot0 side **right** / wrist **left**, and robot1 side
**left** / wrist **right**. Robot1's launcher enables `--mirror_y`: horizontally
flip RGB and negate pose indices `[1,3,5]` before normalization; negate action
indices `[1,3,5]` after unnormalization. Gripper values are unchanged. This uses
the same ideal mirror assumption as dataset conversion. SpaceMouse remains in
the physical robot frame. Existing reset and episode control flow are retained.

`eval.log` on the GPU records episode success/return/length, SpaceMouse override
step counts, and the final success rate (including intervened episodes).
Use a fresh output directory with an existing parent. Videos are written on the
workstation to `EXPO_CLIENT_VIDEO_DIR` and show physical views before flipping.


## Running Real-Time EXPO-FT

At a high control frequency, a VLA forward pass may not fit inside one control step. Real-Time EXPO-FT trains `RealTimeEXPOFTLearner` instead, which

- **performs slow VLA inference asynchronously** -- the next chunk is inferred on a background thread while the robot executes the last `--delay` actions of the current one, and it is inpainted on the executed action prefix so the two chunks join smoothly (RTC-SFT, [arXiv 2512.05964](https://arxiv.org/abs/2512.05964));
- **transforms actions with a fast, reactive edit policy at execution** -- a fast, reactive edit policy takes in the latest observation at the time of exection and predict edits on the VLA action candidates. 

`scripts/dynamic_pick/` is the complete example, on the dynamic pick task: pick up a cube that is dropped at a fresh position after every success. Its client and data steps mirror `scripts/pick/` -- the base policy, the control rate and the delay flags are what differ.

### How the control loop works

`AsyncChunkSampler` (`expo_ft/utils/loop_utils.py`) drives it, identically in the synchronous driver, the async driver and evaluation:

1. once exactly `delay` actions remain in the plan, `sample_pre_cache` starts on a background thread from the *current* observation;
2. the robot keeps executing those `delay` actions;
3. at the replan boundary `sample_actions` finishes the chunk, conditioned on the executed prefix.

`--delay=0` collapses this to a plain blocking `agent.sample_actions` at the boundary, i.e. same as [EXPO-FT](#running-expo-ft). If `replan_steps < delay`, more than one chunk would be in flight -- which the single-slot async path cannot represent -- so the sampler falls back to boundary-synchronous delayed inference reconstructed from the executed history (equivalent, just not overlapped). A human takeover discards the prefix and any in-flight inference.

### Flags

| flag | meaning |
| --- | --- |
| `--delay` | inference latency to hide, in env steps; `0 <= delay <= replan_steps`. Size it as `ceil(latency_ms * control_hz / 1000)`; the dynamic pick scripts use `5` at 30 Hz (~167 ms). |
| `--replan_steps` | action-chunk execution horizon (default 8). Keep SFT, RL and eval aligned. |
| `--sim_latency` | extra artificial inference latency in ms per `sample_actions` call; use it to charge a baseline the same latency budget. |
| `--config.filter_N` / `filter_n_edit` / `filter_temperature` | seed pool scored per backup sample, edit count used in the backup (`<0` = reuse `n_edit_samples`), and seed-selection temperature (`0` = argmax over `Q_f`). |
| `--config.filter_add_delayed_obs` | let `Q_f` also see the actor's delayed observation; only matters at `delay > 0`. |
| `--config.p1_max_delay` | offline RTC-SFT only: prefix length is drawn `d ~ Unif{0..p1_max_delay}` per example (`-1` = `replan_steps`). |

`--delay`, `--replan_steps` and `--sim_latency` exist on `train_pi_robo.py`, `train_pi_robo_async.py` and `eval_droid_policy.py` alike, and must match across the three.

### 1. Finetune pi0.5 with action-prefix conditioning (RTC-SFT)

This is where the two pipelines' pretraining differs. [EXPO-FT](#1-finetune-pi05-sft) runs OpenPI's `train.py`, which always denoises a chunk from pure noise. Here, `train_offline_rtc.py` (`RTCLearner`, `configs/model/rtc_pi_config.py`) trains the same pi0.5 LoRA on the same LeRobot dataset and normalization assets, but each example gets a **clean action prefix** of length `d ~ Unif{0..p1_max_delay}` and the flow-matching loss is applied only to the remaining postfix -- so the prefix-inpainted delayed inference used at deployment is in-distribution.

```bash
# On the server / GPU machine.
bash scripts/dynamic_pick/offline_train.sh
```

Parameters to update in `offline_train.sh`:

- `--dataset_path` / `--num_data` -- collected demonstrations and how many episodes to train on
- `--config.p1_max_delay` -- max prefix length; keep it ≥ the `--delay` you plan to deploy (`-1` falls back to `replan_steps`)
- `--config.freeze_pi05_encoder=False` -- offline SFT trains the encoder as well
- `--config.pi05_assets_dir` / `--config.pi05_asset_id` -- normalization assets from the stats step
- `--output_dir` / `--run_name` / `--max_steps` -- checkpoint location and length; the example runs 4000 steps, checkpointing every 2000

The resulting checkpoint loads through the same `--config.pi05_weight_loader_path` flag, so nothing downstream needs to know which SFT produced it. The example script is a SLURM batch file (`#SBATCH` headers); drop them to run it directly.

Online, the prefix length is *not* drawn at random: it is the deployed `--delay`, and `0` for the first chunk of an episode, so the BC term is trained on exactly the conditioning that runs on the robot.

### 2. Online finetuning with hidden latency

After RTC-SFT, finetune online with `RealTimeEXPOFTLearner`. The server and client communicate over WebSocket: the client (rollout server) runs the environment, the server (learner) runs the RL training loop.

**Start the DROID server on the NUC**:

```bash
# On the NUC, from the DROID install root (for example client/droid
# in this checkout, or the NUC's standalone DROID checkout).
python scripts/server/run_server.py
```

**Start the client** (on the robot machine):

```bash
# On the client / robot machine.
bash scripts/dynamic_pick/run_policy.sh
```

Parameters to update in `run_policy.sh`:

- `SERVER_HOST` -- hostname or IP of the GPU training machine (required; the script exits if unset). Set it in the script or pass it inline: `SERVER_HOST=my-gpu-box bash scripts/dynamic_pick/run_policy.sh`. The client dials the learner directly, no SSH tunnel needed
- `--port` -- learner port; keep aligned with the learner's `client_port`
- `--config_task_path` -- task config for the rollout environment

The learner listens on `client_host`/`client_port` (`0.0.0.0:8103` in the dynamic pick scripts) and waits for the client to connect, so either side can be started first. The robot machine only needs outbound access to that port; if it cannot reach the training machine directly, forward the port with `ssh -L 8103:localhost:8103 <training-machine>` on the robot machine and dial `localhost`.

**Then start the server** (on the GPU training machine):

```bash
# On the server / GPU machine.
bash scripts/dynamic_pick/run_server.sh        # synchronous
bash scripts/dynamic_pick/run_server_async.sh  # asynchronous
```

> **Async training:** Requires ≥ 2 GPUs (1 sampler + ≥ 1 updater). Use it when one episode takes a long time; otherwise synchronous training can yield better results.

Key parameters to configure in `run_server.sh` and `run_server_async.sh`:

- `dataset_path` -- path to the collected demonstration data
- `num_data` -- max offline demo episodes to seed into the replay buffer (0 = all)
- `--delay` -- inference latency to hide, in env steps (see [Flags](#flags))
- `--config.pi05_weight_loader_path` -- the RTC-SFT checkpoint from step 1
- `update_type` / `num_updates` -- for synchronous training, recommend: use episode updates with `env_steps / num_updates` close to 20-30
- `step_interval` -- alternative to a fixed `num_updates`: one gradient update per this many collected transitions
- `edit_scale` -- edit scale
- `client_host` / `client_port` -- interface and port the learner listens on; `0.0.0.0` accepts the client from any machine

> **Client recovery:** If the client hits an error or the robot gets stuck during online training, you can stop and restart only the client. The server waits for the policy/environment connection to recover, then continues training once the client is restarted.

### 3. Evaluation at the deployed delay

```bash
# On the server / GPU machine.
bash scripts/dynamic_pick/eval_policy.sh
```

Use the same `--delay`, `--replan_steps`, model config and `filter_*` settings as training; `--checkpoint_dir` points at the run from step 2.


## Citation

<!-- TODO(release): fill in the author list and arXiv id below. -->

```bibtex
@misc{dong2026reinforcement,
      title={Reinforcement Learning for Real-Time Vision-Language-Action Policies}, 
      author={Perry Dong and Kuo-Han Hung and Dorsa Sadigh and Chelsea Finn},
      year={2026},
      eprint={2609.18207},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2609.18207}, 
}
```


```bibtex
@misc{dong2026expoft,
      title={EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models},
      author={Perry Dong and Kuo-Han Hung and Tian Gao and Dorsa Sadigh and Chelsea Finn},
      year={2026},
      eprint={2605.25477},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.25477},
}
```
