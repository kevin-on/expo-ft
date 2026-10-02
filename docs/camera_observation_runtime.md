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
