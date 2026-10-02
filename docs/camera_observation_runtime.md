# Workstation camera observations

Robot JSONs select physical camera serials and eyes with `side_camera_id` and
`wrist_camera_id`. Only those eyes are retrieved for online/eval. Collection can
explicitly request both eyes; its stored pose/actions remain physical coordinates.
Policy view serials also restrict physical camera initialization. If
`camera_serials` is supplied, only its intersection with the selected views is
opened; this preserves the exclusion of virtual/blank cameras. Without a view
selection, collection retains its explicit camera list and both-eye settings.
`camera_kwargs.hand_camera` / `varied_camera` accept `capture_resolution`
(`720p`, `1080p`) and `camera_fps` (15/30; 720p also 60). Defaults in the reader
remain 1080p/15 for callers without explicit configuration.

`camera_crops[serial_eye][capture_resolution]` is `[x,y,width,height]` in source
pixels, applied before model resizing. Missing resolution entries mean no crop.
Intrinsics are shifted and scaled accordingly. Raw recordings remain uncropped.

Configured 720p side ROIs come from the retained 20261002 rectified SDK
calibration comparisons. Wrist ROIs [158,88,960,540] come from the local ZED
factory calibration files SN15577469/SN12841040 (HD vs FHD intrinsics).
They have NOT been validated against rectified live wrist images; verify FOV
alignment when camera access is authorized. No camera was opened to derive them.

DROID is a separate ignored checkout. This feature requires its companion
`camera-observation-runtime` branch as well as this EXPO branch. Do not deploy
only the parent repository while leaving an older DROID checkout installed.

## Model coordinates live at the WS boundary

`model_frame.mirror_images.side/wrist` independently flip the selected model RGB
inputs horizontally. `model_frame.mirror_robot_coordinates` reflects Y/roll/yaw
for both Cartesian observations and policy actions. Configure all three true for
robot1 and false for robot0 to reproduce the existing mixed dataset convention.
All three booleans are required at WS environment creation, including explicit
false values. Missing/partial settings are rejected before constructing the
environment; old robot JSONs cannot silently disable robot1 reflection.
`ModelFrame` owns the live transformation; ILIAD consumes already canonical data.
Human commands execute physically, then their actual/clipped actions are reflected
back for replay. Reset, bounds, raw video and collection HDF5 stay physical.
Collection's offline converter still owns conversion of its physical records.

New WS/inference endpoints negotiate `ws-model-frame-v1` before environment
construction. Update both endpoints together; old binaries must not silently
apply their own mirror or omit the WS transform. `eval_droid_policy --mirror_y`
is superseded by the WS robot JSON and now rejects a true value.

## Latest-frame mode

`camera_buffer: {"enabled": true, "max_age_ms": 250, "timeout_seconds": 10}`
uses one producer per robot, a persistent parallel reader pool, and one latest
complete side/wrist slot. Crop/resize runs before publishing. Observation reads
reuse those prepared images but read robot state freshly; images are not mutated
once published. Episode video retains the raw images selected by observations,
not every background frame. The polling rate follows the slowest configured
camera FPS. Reused frames are permitted and identifiable by SDK timestamps.

Reset sets a minimum SDK frame timestamp; no pre-reset frame passes that barrier.
Missing, stale, future-dated or failed frames surface an error rather than silently
running on old pixels. SDK timestamps are host-reception time, not sensor exposure
start. Hardware mode changes/recording stop the producer before touching the SDK.
If one parallel camera read fails, the producer waits for every submitted read
to finish before exiting. A stuck read makes shutdown time out without allowing
a concurrent SDK mode change or camera release.
SVO recording reads synchronously. Collection also disables this cache explicitly.

## Frame-to-action timing

Observation RPCs carry `observation_metadata` alongside (never inside) model
inputs: side/wrist `frame_received_ms`, `selected_ms`, and `buffer_sequence`.
These are SDK IMAGE timestamps in WS Unix milliseconds, not ILIAD timestamps or
estimated exposure times. Online rounds and coordinated eval retain the metadata
of the observation used to create each action chunk, even while newer observations
arrive. Every action from that chunk echoes the same metadata to WS.

WS samples `last_action_send_ms` after physical clipping, just before the DROID
update RPC. The difference is host frame reception → WS robot-command dispatch;
USB capture latency and NUC/controller application latency are not included.
`[timing][frame action]` logs contain separate side/wrist ages and action source.
Eval episode step timings also contain `action_frame_timing`. Human actions and
zero handoff polls have no policy-frame age (they are not model decisions).
Observation timing includes side/wrist age at buffer selection. Missing timestamps
stay missing rather than being substituted with a recent unrelated frame.

Legacy single-robot asynchronous/RTC entrypoints still use the common WS coordinate
boundary, but do not claim chunk provenance timing until their asynchronous plans
explicitly carry this metadata; the split/colocated round and coordinated eval
paths implement it.

## Shared rollout logs and TUI

Coordinated eval (direct and receiver) and split/colocated Online FT use
`expo_ft/env/rollout_timing.py` for step records, live metrics and formatting:

`RolloutMetrics` owns both Hz history and frame ages, without a separate rate
class. Online delivers each completed step's transition and timing together via
`on_transition(robot, step, transition, timing)`; the runner sends only the
transition to replay and passes timing directly to the dashboard.

```text
9.8 Hz | Frame age S/W: 120/125 ms
```

- Hz is recent completed **control steps**, across ten intervals, including
  inference, RPCs and control-period waits. It is not camera or pure inference Hz.
  It decays during stalls and resets at the next episode/round.
- S/W are the latest completed action's side/wrist frame ages at WS command
  dispatch. They retain the action chunk's original observation. They do not
  increase on TUI refresh and are not the next observation's buffer-selection age.
- Human control still counts for Hz, but has no policy frame age. Missing
  timestamps and handoff polls display `—`; all metrics are hidden while idle,
  resetting or waiting for an update.

Both loops emit `[timing][rollout step]` JSON records with mode, robot, episode
(eval) or round (online), step, planning time, action RPC time, observation RPC
time, observation breakdown and WS action frame timing. `plan_ms` includes the
policy queue/lock wait. `observation_ms` measures the **next** observation after
this action; `action_frame_timing` refers to the frame used for this action.
RPC intervals use inference-host monotonic time; WS computes frame ages locally.
No cross-machine timestamp subtraction is used.

Eval keeps these same step records in `episodes.jsonl` as well as its eval log
(`logs/eval.log` for receiver eval, `eval.log` for direct eval). Online uses the
existing inference/`--rollout_log` destination. Telemetry is not inserted into
replay or sent to the learner. This adds no RPCs or device reads and does not
change action selection, rollout ordering or checkpointed training metrics.

## Companion revisions and device-free checks

DROID companion branch: `camera-observation-runtime`. Deploy its current checkout,
including the camera-selection and read-draining fixes, with this EXPO checkout.
OpenPI is unchanged (validated companion `590fa99`). The feature worktree's
client environment/source links are local conveniences, not tracked deployment
inputs. Use the new EXPO and DROID sources together; a main-worktree venv may have
an editable DROID path pointing to the old source, so explicitly set PYTHONPATH.

```bash
source /scr/kevinon/env.sh
export PYTHONPATH="$PWD/client/droid:$PWD:$PWD/tests"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
client/.venv/bin/python -m unittest discover -s client/tests -p 'test_*.py'
client/.venv/bin/python -m unittest tests.test_robot_eval tests.test_split_reset_overlap_isolated tests.test_async_video_isolated tests.test_rollout_dashboard_isolated
PYTHONPATH="$PWD/tests/cpu:$PYTHONPATH" client/.venv/bin/python -m unittest test_sft_eval test_mirror_online
```

Validation on 2026-10-02: 100 client tests, 49 eval/reset/video/dashboard tests,
and 7 SFT/online converter parity tests passed without devices or GPUs. Synthetic
SDK tests cover selected-eye reads, capture settings, and retrieval failure.
They also check unused-camera exclusion and that a failed parallel read cannot
permit a mode change while another read is active. WS creation tests reject
missing/partial mirror settings before hardware construction.
Synthetic RPCs check policy/human actions and timestamps; simulated rounds check
chunk provenance for one/two robots. This establishes correctness of those paths,
not actual camera throughput or physical control latency. Camera access remains
prohibited until explicitly authorized.

Subsequent shared-telemetry validation: 101 client tests and 46 targeted
eval/round/dashboard/formatter tests passed (147 total). The full remote eval
RAM suite could not pass in the WS client interpreter because it lacks Linux
sealing constants; its receiver TUI formatting test passed independently. See
`tests/README.md` for isolated test dependencies and the focused command.

Read-only selection preview (does not enumerate or open cameras):

```bash
client/.venv/bin/python scripts/multi_robot/benchmark_camera_latency.py \
  --dry-run --resolution 720p --fps 30 --camera-count 4 --selected-eyes
```

`benchmark_camera_latency.py` keeps three raw-read scheduling modes: default
per-robot processes (side/wrist read concurrently), `--all-parallel` (one process
and read pool for all selected cameras), and `--independent-cameras` (one process
per camera; only phase starts synchronize). These last two flags are mutually
exclusive. All modes construct normal per-camera DROID wrappers, pass native
`camera_views` for `--selected-eyes`, and join reads before closing cameras.
Resolution/FPS overrides are in memory only. No buffer, crop or mirror is enabled;
this measures raw capture/retrieval, not buffered rollout observation latency.
`tests/test_camera_benchmark.py` checks grouping, settings and failure cleanup
without an SDK or devices. Existing `pair_ms` fields mean the entire worker's
read group (one, two or four cameras, depending on the selected mode).
