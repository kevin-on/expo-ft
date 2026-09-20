# TPUQ smoke test

From the configured TPUQ controller checkout:

```bash
.venv/bin/tpuq submit /path/to/expo-ft/scripts/multi_robot/tpu_smoke.sh \
  --repo git@github.com:kevin-on/expo-ft.git \
  --commit FULL_PUSHED_COMMIT_SHA --name expo-two-robot-pi05-smoke
```

The job reserves four physical chips in `usc2`, prepares a Python 3.11/JAX 0.5.3
TPU environment, and pins OpenPI to `46407a41183b037313a383ff679683f2773b766d`.
It downloads the public `pi05_base` checkpoint to the worker cache. No robot SDK
or hardware is required. The requirements cover the JAX path used by this test;
they are not a replacement for the full workstation/client environment.

The test calls the actual `train_pi_robo.main`, with two synthetic WebSocket
clients returning 224x224 images, Cartesian state, and 16/24-step episodes.
It uses the full pi0.5 LoRA configuration and existing EXPO losses, batch size 8,
UTD 1, two base and two edited candidates, two Q functions, and offline ratio 0.5.
The demo and identity normalization statistics are generated fixtures and must
not be used to control a physical robot.

Assertions cover real finite inference, an update only after both episodes end,
finite update metrics, changes to actor/critic/edit-actor parameters, and refreshed
inference parameters in the following round. Eight rounds produce 16 episodes,
320 online transitions, and three updates after the existing warmup. The final
model checkpoint, round ledger, and 128/192 per-robot transition files are checked.
Checkpoint restore, task success, production-scale batches, and physical robot
timing are outside this smoke test.

Small results and the installed dependency versions are uploaded to
`gs://soe-iris-kevin-usc2/expo-ft/smoke/TPUQ_TASK_ID/`. The checkpoint and generated
images remain in the worker attempt directory under `logs/tpu-smoke/`.
