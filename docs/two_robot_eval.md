# Coordinated SFT and online evaluation

`eval_sft_robots.py` owns one GPU model and a terminal dashboard. It supports
robot0, robot1, or both. Model loading is described in [model_config.md](model_config.md).
No checkpoint-config branch changes were merged.

1. Allocate/stage the matching GPU runtime and source as in the split operating guide.
   Expose one GPU. Use an interactive GPU terminal (`ssh -tt` across SSH).
2. Set a completed checkpoint step, a fresh GPU output directory, and an absolute
   workstation video directory. For SFT:

   ```bash
   export SFT_CHECKPOINT=/path/to/4999
   export EXPO_EVAL_OUTPUT_DIR=/path/to/new-eval
   export EXPO_CLIENT_VIDEO_DIR=/scr/kevinon/workspace/expo-ft-fork/data/videos/new-eval
   bash scripts/pick/eval_sft_policy.sh both
   ```

   For online checkpoints, set `ONLINE_CHECKPOINT=/path/to/checkpoints/6400` and
   use `bash scripts/pick/eval_policy.sh both`. If the initial SFT assets moved,
   set `INITIAL_SFT_CHECKPOINT`; their config and normalization must match.
3. Establish dedicated forwards for ports 8202 and 8203 to this GPU node if needed.
   Existing authenticated SSH connections may be reused. Match `EXPO_EVAL_BASE_PORT`
   on both hosts when overriding it.
4. **When ready to open hardware/reset**, run on WS:

   ```bash
   export EXPO_EVAL_HOST=127.0.0.1  # with the above dedicated tunnels
   bash scripts/pick/run_sft_eval_client.sh both
   ```

   `0` or `1` selects one robot on either script. For `both`, the client launcher
   starts two child processes and records their logs under `data/logs/eval-*`.
   It only stops its own children when one exits or the launcher is interrupted.

WS MP4 encoding defaults to **2 threads per encoder**. Override it on the client:

```bash
bash scripts/pick/run_sft_eval_client.sh both --video-encoder-threads 1
# Online rollout client uses the same option:
bash scripts/multi_robot/run_workstation_rollout.sh 0 --video-encoder-threads 1
# Data collection (also accepts --video_encoder_threads):
ROBOT_ID=0 bash scripts/pick/collect_data.sh --video-encoder-threads=1
```

The value must be positive. It controls libx264 encoding, including asynchronous
episode videos and the optional dedicated record camera. It does not limit image
preprocessing or total process CPU use. Concurrent writers each get this many
threads; collection can have several writers per robot. Fewer threads can delay
saving and cause the bounded recording queue to wait. Restart the WS client or
collection process to apply a changed value; the GPU server needs no restart.

The GPU terminal shows each robot's status, step count, completed episodes,
success count and last success/failure. Each robot resets independently. Once
**all selected robots are ready**, Space starts one round. Space while busy is
ignored. No start requests are queued. The first observation is read after Space;
both first plans must be ready before either first action is dispatched. A shared
lock serializes policy/RNG use; the model is not duplicated. This barrier is not a
hard real-time simultaneous-motion guarantee.

Each finished robot resets immediately. The next round again waits for all ready
and another Space. `r` / `t` repeat reset for READY robot0 / robot1, respectively.
The selected robot shows resetting, then returns to READY without adding an
episode or starting rollout; reset/start keys for busy robots are ignored.
`q`/Ctrl+C closes this eval's connections; errors do not silently
recreate robot environments. An RPC already executing on WS cannot be recalled.
The WS client validates camera views and task frequency/episode length/action
format before constructing the environment. Existing single-client eval retains
its client-local pause behavior.

`episodes.jsonl` is flushed after each completed episode and contains robot ID,
round, success, steps, human steps, per-episode intervention rate, `had_intervention` and `success_without_intervention`, plus observation/inference/action
RPC timings. `eval_config.json` records checkpoint/task/eyes/omitted cameras.
Each step's `timings[].observation_breakdown` additionally records local RPC,
request packing/send, response wait/unpack durations and response byte count.
Its `ws` member contains observation processing, termination (including terminal
video submission/backpressure), and response packing durations. `ws.environment`
splits observation construction into raw observation, video-frame preparation,
and image transformation; robot-state and combined camera reads are nested
within raw observation. Side/wrist `*_grab_ms` use existing SDK read timestamps
and exclude image retrieval/resize. They overlap because cameras read in parallel;
do not sum them or add nested fields to their parent durations.
`transport_and_queue_ms` is RPC time minus request packing, response unpacking,
WS processing and WS packing. It includes transport, scheduling, socket queues,
and the final WS serialized-buffer copy; it is not pure network latency. Packing
is measured once before appending the small timing entry to the MessagePack map.
All other durations use local monotonic clocks, without cross-host timestamp
subtraction. Timings are response metadata, never policy inputs. No flag is
required; older WS clients simply omit the WS breakdown. Restart both the WS
client and eval server to apply instrumentation to a new run.
Video-frame preparation (color conversion/concatenation) and MP4 encoding are
asynchronous on WS and drained when the environment closes. During rollout we
retain only selected camera images already copied out of SDK storage; no second
copy/concatenation is made for video in the control loop. At episode end the
worker prepares each frame in place in the detached buffer, then encodes it.
`video_frame_ms` therefore measures buffer bookkeeping for asynchronous eval;
`VIDEO_PREPARE_FINISHED` in the WS log records background preparation time.
The existing one-pending-episode bound remains: if the prior video is still
saving when the next episode ends, submission waits instead of growing memory.
Unfinished episodes are not reported as completed successes/failures.

The eval and online inference TUIs show each robot's live rollout Hz over up to
the latest ten completed-step intervals. This measures control-loop throughput
(including observation/RPC/replanning), not GPU inference calls per second.
It updates after two completed steps, decreases when a running step stalls,
and shows `—` during ready/reset/update states. Every new episode clears the
window, excluding manual waiting and reset time. No CLI option is required.

`tests/test_robot_eval.py` uses fake environments to verify gates, independent
reset, shared policy serialization, mirroring, termination and failure cleanup.
`tests/gpu/robot_eval.py` exercises the same coordinator with a real checkpoint
and recorded observations on an allocated GPU; it never connects to the WS.

Validation on 2026-09-29: five coordinator tests, six async-video tests, five
isolated mirror/eval tests, and fourteen RPC/reset-pause tests passed. On ILIAD
H200, the real wrist-off SFT policy and a newly saved online EXPO checkpoint each
completed one two-robot round against recorded/fake environments. This verifies
software coordination and model loading, not physical timing or task success.
