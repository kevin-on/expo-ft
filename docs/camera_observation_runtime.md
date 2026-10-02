# Workstation camera observations

Robot JSONs select physical camera serials and eyes with `side_camera_id` and
`wrist_camera_id`. Only those eyes are retrieved for online/eval. Collection can
explicitly request both eyes; its stored pose/actions remain physical coordinates.
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
