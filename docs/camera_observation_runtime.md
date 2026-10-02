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
