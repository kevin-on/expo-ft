# Four-camera HD1080 / 15 FPS validation

On 2026-09-21, the workstation completed a **300-second simultaneous capture**
with two processes, each owning a side+wrist pair. A ten-second warmup followed
opening all four cameras. The camera SDK confirmed 1920x1080 at 15 FPS for every
camera. No RobotEnv, robot control, learner connection or video recording was used.

| Pair | Role | Serial | Measured FPS | Frames | Max timestamp interval |
| --- | --- | --- | --- | --- | --- |
| 0 | side | 38651013 | 14.9289 | 4479 | 68 ms |
| 0 | wrist | 15577469 | 14.9289 | 4479 | 69 ms |
| 1 | side | 29838012 | 14.9306 | 4480 | 68 ms |
| 1 | wrist | 12841040 | 14.9306 | 4480 | 70 ms |

All four had zero read errors, duplicate/backward timestamps, timestamp-inferred
missing frames and SDK drop-count increases during the measurement window.
SDK drop counters were nonzero at measurement start (49, 2, 54, 7 respectively),
from opening/warmup; these are not reported as drops during the five-minute test.
Actual image timestamps progressed at about 66.986 ms/frame, explaining the
observed 14.93 rather than exactly 15.00 FPS without evidence of skipped frames.

The existing DROID ZedCamera reader retrieves and copies each full left image.
Each pair reads concurrently using a thread pool. Images are discarded after
shape validation; CSV files retain timing, image timestamps, errors and SDK drop
counts. No resized policy preprocessing or video encoding is included. Persistent
pair threads avoid per-poll thread-pool creation used by MultiCameraWrapper;
this tests its camera-reading operation and concurrency layout, not every rollout
CPU operation. Read-call timing includes waiting for the next frame and image
retrieval/copy, and is not sensor-to-action latency.

The SDK logged **NEURAL depth mode**. The existing DROID `depth=False` flag skips
retrieving depth images but does not disable SDK depth computation. This test
preserved that production initialization; no depth-mode optimization was applied.
SDK also emitted self-calibration scene warnings on the side cameras during
startup. Frame throughput does not validate scene coverage, image quality or
geometric calibration.

Workstation resource samples after startup showed about 22.2% mean aggregate CPU
usage and roughly 39–40% GPU use; detailed values are in `resource-summary.json`.
These include background desktop usage. Before the test GPU use was 7% / 720 MiB.
USB topology was unchanged: all four video devices enumerated at 5 Gbit/s on bus 2,
with two cameras and a USB Ethernet adapter sharing an external hub. This result
covers that topology under the measured load, not arbitrary future USB traffic.

All cameras were closed at completion. Production camera FPS defaults, robot JSON
mappings and the running learner were not changed by this test.

## Reproduce

From `/scr/kevinon/workspace/expo-ft-fork`, after checking no other application
owns the cameras:

```bash
source /scr/kevinon/env.sh
client/.venv/bin/python scripts/multi_robot/test_camera_capture.py \
  --pair 38651013 15577469 --pair 29838012 12841040 \
  --fps 15 --duration 300 --warmup 10 \
  --output /scr/kevinon/data/camera-tests/NEW_RUN_DIRECTORY
```

Use a new output directory. Passing only one `--pair` can isolate a two-camera
setup. The script changes FPS/resolution in its spawned processes only and never
edits the DROID file. Its explicit acceptance criteria are recorded in
`request.json`: full duration, >=98% target FPS, <=1% timestamp-inferred loss,
zero read errors/duplicate/backward timestamps and maximum frame gap <=250 ms.
The SDK dropped-frame counter is also recorded separately.

Evidence directory:
`/scr/kevinon/data/camera-tests/1080p15-four-20260921/`.
Primary results: `summary.json`, `pair-*-frames.csv`, `pair-*-summary.json`,
`resources.jsonl` and `resource-summary.json`.


## Production defaults updated; depth NONE follow-up

On 2026-09-21, the local DROID fork and deployed workstation checkout were
updated to request HD1080 / 15 FPS by default and disable SDK depth computation
when neither depth nor pointcloud output is requested. Robot 1's wrist mapping
now uses serial 12841040 and has no blank cameras. Depth requirements changing
through set_reading_parameters invalidate initialization for the next mode call.
Five existing camera-related offline DROID regression tests passed.

At the user's request, the follow-up measurement was limited to **30 seconds**
after ten seconds of warmup. `--use-camera-defaults --expect-depth-none` ensured
that the actual production defaults were tested without overriding FPS/resolution,
and all four SDK instances reported `DEPTH_MODE.NONE`.

| Serial | Received FPS | Read errors | Timestamp-inferred missing | SDK drop increase | Max frame interval |
| --- | --- | --- | --- | --- | --- |
| 38651013 | 14.9032 | 0 | 0 | 0 | 68 ms |
| 15577469 | 14.9365 | 0 | 0 | 0 | 68 ms |
| 29838012 | 14.9468 | 0 | 0 | 0 | 68 ms |
| 12841040 | 14.9468 | 0 | 0 | 0 | 68 ms |

No duplicate or backward timestamps were observed. Two initial camera opens
needed one internal SDK retry each and then succeeded; this was before warmup
and is distinct from the zero streaming errors during measurement. Cameras were
released at completion and both robot camera/HID mappings passed enumeration.

Three steady capture samples showed workstation GPU use **18%**, memory
**1547 MiB**, aggregate CPU **18.93–19.10%**. Compare the earlier five-minute
NEURAL run's roughly 39% / 2355 MiB. These are whole-workstation samples from
different-duration runs, not a controlled isolated GPU kernel benchmark.

Command (use a new output directory on rerun):

```bash
client/.venv/bin/python scripts/multi_robot/test_camera_capture.py \
  --pair 38651013 15577469 --pair 29838012 12841040 \
  --fps 15 --use-camera-defaults --expect-depth-none --duration 30 --warmup 10 \
  --output /scr/kevinon/data/camera-tests/NEW_DEPTH_NONE_RUN
```

Evidence: `/scr/kevinon/data/camera-tests/1080p15-depth-none-30s-20260921/`.
No rollout client, controller initialization or robot reset was started by this
follow-up. The next step is the actual two-robot end-to-end rollout.
