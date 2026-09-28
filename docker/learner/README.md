# Learner container build and validation

For the current GH200/H200 deployment and prebuilt image locations, start with
[the operating guide](../../scripts/split/README.md). These Docker/Apptainer build
recipes are for x86-64; use the ARM recipe described in
`/scr/kevinon/workspace/expo-ft-tools/README.md` for GH200. Do not rebuild a working
image or create a new environment just to launch another run.

The reusable [Delta H200 tools](delta/README.md) are a separate, base-model
validation route. They are not the current two-host online training launcher.
Source and companion OpenPI must be deployed explicitly; old baked-in source is
not updated by a workstation Git pull. Build only on a suitable compute host.

## Build

Use the EXPO repository as the build context, including the pinned OpenPI checkout
at `expo_ft/agents/vla/openpi` (commit
`19c1b33951bff0a80782a4d21cb552d641555d3f`). The Dockerfile-specific ignore file
excludes robot client files, credentials, virtual environments, and model data.

```sh
docker build -f docker/learner/Dockerfile -t expo-ft-learner:jax053 .
```

Build and execute on an allocated compute node. Keep the container storage and
runtime caches on node-local storage. Never mount an NFS Python environment over
the image's installed libraries. Keep Slurm's allocated GPU visibility and limits.

## Validation

The harness loads a recorded HDF5 episode through the actual EXPO dataset/replay
code and builds the complete EXPOLearner. N=8, n_edit_samples=8, num_qs=10, image
resolution, LoRA configuration and critic architecture retain the production
config. Small-batch smoke testing uses batch_size=8, UTD=1; a separate process
must use batch_size=64, UTD=20 to assess the example training defaults.

Inside the container, bind the recording's episode-parent directory to `/demo`,
base params to `/params`, the OpenPI download cache to `/openpi-cache`, and a
node-local writable directory to `/output`. Set OPENPI_DATA_HOME=/openpi-cache,
WANDB_MODE=disabled, and XLA_PYTHON_CLIENT_PREALLOCATE=false for measurement.

```sh
python docker/learner/verify_assets.py /openpi-cache
python tests/gpu/learner_smoke.py --dataset /demo --params /params \
  --output /output/small --batch-size 8 --utd-ratio 1 --updates 3
python tests/gpu/learner_smoke.py --dataset /demo --params /params \
  --output /output/small --phase restore --batch-size 8 --utd-ratio 1
python tests/gpu/learner_smoke.py --dataset /demo --params /params \
  --output /output/default --batch-size 64 --utd-ratio 20 --updates 3 --skip-checkpoint
```

The first process checks finite inference and losses, changes in VLA LoRA,
critic, edit-actor and image encoder parameters, and checkpoint save. The second
process restores and checks parameter hashes, step count and deterministic action
reproduction. JSONL stage records include JAX device allocation/peak counters;
also sample nvidia-smi externally to record process/device memory. First-call
compilation and subsequent execution times are recorded separately.

Normalization is computed from the test recording with finite ranges for
constant dimensions. These are validation fixtures, not approved deployment
statistics. Base weights plus initialized LoRA test execution and memory only;
they do not establish policy quality or real-robot latency. No robot connection
or motion occurs. Archive the test output and installed dependency versions.
