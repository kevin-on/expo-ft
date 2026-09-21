# EXPO learner/inference container validation

This image contains the JAX/CUDA learner and inference stack only. It does not
contain the workstation robot SDK, credentials, model weights, or recordings.
It uses the JAX 0.5.3 / Flax 0.10.2 / Optax 0.2.4 versions of the existing TPU
validation environment, with CUDA instead of TPU. This explicit runtime avoids
using the unrelated client venv and does not modify the root learner lockfile.
It does not validate the separate LeRobot conversion / offline SFT pipeline.

## Scope of these scripts

**For Delta H200, use [the reusable Delta launcher](delta/README.md) and its
[validation results](delta/VALIDATION.md).** The `delta/` scripts accept input and
output paths and run the current checkout inside the validated SIF.

The other scripts here preserve the iris-ws-5 experimental setup. They intentionally keep
workstation paths and Stanford Iris/Iliad Slurm accounts, partitions and storage
paths so later experiments can use the tested commands as references. They are
not portable cluster defaults: check GPU hosts, resource requests and filesystem
paths before running on another server, including Delta or DeltaAI. The existing
container targets x86-64; DeltaAI needs an ARM-compatible environment.

`real_robot_17539504/` is a historical job snapshot. Its job ID, learner host,
WandB run IDs and resume/output paths refer to that experiment; update them for
a new run. Keep credentials outside the repository. The dated validation results
below describe completed attempts, not current node health or allocation status.

## Build

Use the EXPO repository as the build context, including the pinned OpenPI checkout
at `expo_ft/agents/vla/openpi` (commit
`46407a41183b037313a383ff679683f2773b766d`). The Dockerfile-specific ignore file
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

## Cluster execution and current validation status (2026-09-20)

The user approved running a Docker-based image with Apptainer because direct
Docker daemon access is unavailable. `learner.def` uses the same Python 3.11
Trixie base and learner requirements as the Dockerfile. Trixie is needed for the
cluster's host fakeroot library (Bookworm failed on missing GLIBC_2.38).

- Build job **17530109** succeeded on iris4. `uv pip check` passed for 134
  packages, and the actual `train_pi_robo` / EXPOLearner imports passed.
- Image: `/iris/u/kevinon/artifacts/expo-ft/expo-ft-learner-jax053.sif`.
- Image SHA256: `3ca61c8fb26fab7597b01c22ca1e5599657a59ae155117010e0185ac11f2f7ef`.
- `requirements-tested.txt` records the exact installed versions in that image.
- GPU job **17530222** on iris9 verified all 21 weight/tokenizer files against
  GCS CRC32C and the transfer manifest's SHA256 (12,445,985,954 bytes).
- CUDA initialization then failed with `CUDA_ERROR_ECC_UNCORRECTABLE` on the
  allocated L40S, before loading the model. This is not an out-of-memory result.
  See the subsequent A40 results below; this L40S attempt did not reach model loading.
- Job **17530244** was cancelled when the user switched validation to one A40.
- The read-only iris9 check (**17530297**) found DRAM uncorrectable ECC counts
  of 2 volatile / 9 aggregate and pending row remapping. Kernel log access was
  denied, so the time of the original error is unknown.

The GPU job stages the SIF, weights and recording on node-local storage. It checks
CUDA before staging the weights, drops NFS-specific permission metadata while
copying contents, and retains logs/results on shared storage. It uses exactly one
L40S and does not access robots or restart/reset any GPU. Jobs:

```sh
# On scdt; files prepared under this directory.
ssh sc-codex 'sbatch /iris/u/kevinon/workspace/expo-ft-docker-validation/build_iris.sbatch'
ssh sc-codex 'sbatch --exclude=iris9 /iris/u/kevinon/workspace/expo-ft-docker-validation/test_iris.sbatch'
```

Build outputs and GPU result archives:
`/iris/u/kevinon/outputs/expo-ft-docker-validation/`.
Public weights and download manifest:
`/iliad/u/kevinon/artifacts/expo-ft/openpi-cache/`.
Test recording: `/iris/u/kevinon/artifacts/expo-ft/demo/0/traj.hdf5`.
The source of this recording is the workstation's 226-step teleop-test episode.
Successful learning or task completion is not assumed.


## A40 follow-up (2026-09-20, completed with capacity limits)

`test_a40.sbatch` requests one A40 in the `iliad` account/partition and stores
results under `/iliad/u/kevinon/outputs/expo-ft-docker-validation/`. It stages
all runtime inputs on local disk and checks for at least 60 GiB and 100000 free
inodes before starting. The SIF and recording remain single-file inputs from
`/iris`; the Python environment always runs inside the local SIF.

- **17530379** failed before CUDA: iliad6's shared local `/tmp`/`/scr` disk was
  full (140 KiB available). Subsequent submissions exclude iliad6. iliad5 had
  4.7 TiB available. No other users' files were removed.
- **17530513** exposed an invalid combination of GNU df options in the new
  scratch check. The script now uses `df --output=iavail` without `-i`.
- **17530665** passed model creation and repeated inference, then ran out of
  memory at its first batch-8 / UTD-1 update with JAX's default 75% pool limit.
- **17530754**, with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, passed the first
  batch-8 update: finite metrics and changed actor/critic/edit-actor/encoder
  parameters. Its second update ran out of memory. This is not a successful
  repeated-training result. Warm inference was about 0.20–0.21 s per 16x7 chunk,
  excluding robot/network overhead.
- **17530842** tested batch 1 but hit an existing singleton-batch shape error:
  `sample_batch_actions` squeezes the batch dimension from critic images and
  interprets the image height (224) as batch size. Production code is unchanged;
  singleton batches are not validated by this harness run.
- **17530983** retries the full validation with batch 2 / UTD 1, three updates,
  and a 95% JAX pool limit. Its second update also ran out of memory.
- **17531495** used batch 2 / UTD 1 and a 98% JAX pool limit. All three updates,
  inference, checkpoint save and independent-process restore passed. Every
  update had finite metrics and changed the four trainable parameter groups;
  restored parameter hashes, step count and deterministic actions matched.
- The separate batch-64 / UTD-20 profile in that job failed during first-update
  compilation/autotuning with a 1.89 GiB allocation error. The job's historical
  Slurm COMPLETED state reflects the old script's optional-profile handling,
  not success of the default configuration. `run-status.txt` records its exit 1.
  The current script now propagates default-profile failures to Slurm.
- Successful batch-2 peak live JAX allocation was **42.88 GiB**. nvidia-smi
  includes reserved pools and CUDA overhead; see `VALIDATION.md` for measured
  device usage and timing. Updates 2/3 took 162/159 seconds in this harness,
  so this validates execution, not practical online-training throughput.

Use `--export=ALL,EXPO_VALIDATE_DEFAULTS=0` with sbatch to run only the validated
small-batch profile. The default remains to attempt both profiles. That run did
not test full robot/WebSocket training or multi-GPU behavior; see the later
two-GPU result below.

```sh
ssh sc-codex 'sbatch --exclude=iliad6 /iliad/u/kevinon/workspace/expo-ft-docker-validation/test_a40.sbatch'
```

## Two-A40 FSDP submission (2026-09-20)

Job **17533545** requests two A40 GPUs on one node, 12 CPU threads and 96 GiB
host memory. Global batch is 8 (4 per device), UTD is the production default 20,
and `fsdp_devices=2`. It checks inference, three updates and independent-process
checkpoint restore. No batch-64 profile is included in this submission.
`test_a40x2.sbatch` excludes no node internally; this submission excludes iliad6
because its local scratch was previously full. It failed at the first update:
inference had committed the RNG to GPU 0, incompatible with the two-GPU learner
state. The corrected run below passed.

The validated SIF is unchanged. A versioned harness snapshot is copied onto
node-local storage and bound read-only over its image counterpart. The harness
checks both GPU visibility and actual partitioning of actor parameters; the
preflight performs a reduction across a two-GPU sharded array. Its SHA256 is
included in the output. Global batch and device counts must be divisible.

Submission snapshot:
`/iliad/u/kevinon/workspace/expo-ft-docker-validation/launches/a40x2-20260920-v1/`.
The script must run with that directory as its working directory because it
contains `learner_smoke.py` beside `test_a40x2.sbatch`.

```sh
ssh sc-codex 'sbatch --exclude=iliad6 --chdir=/iliad/u/kevinon/workspace/expo-ft-docker-validation/launches/a40x2-20260920-v1 /iliad/u/kevinon/workspace/expo-ft-docker-validation/launches/a40x2-20260920-v1/test_a40x2.sbatch'
```

## One-H200 submission (2026-09-20)

Job **17533604** adds one H200 while keeping A40 job 17533545 queued. It requests
12 CPU threads, 128 GiB host memory and 90 minutes. The full default profile is
batch 64 / UTD 20 / three updates, followed by checkpoint restore in an
independent process. JAX uses one device, FSDP 1 and a 95% allocator pool limit.
The same validated SIF, public weights and recording are reused, with local
staging and a read-only harness snapshot. The outcome is pending.

Submission snapshot:
`/iliad/u/kevinon/workspace/expo-ft-docker-validation/launches/h200-20260920-v1/`.

```sh
ssh sc-codex 'sbatch --chdir=/iliad/u/kevinon/workspace/expo-ft-docker-validation/launches/h200-20260920-v1 /iliad/u/kevinon/workspace/expo-ft-docker-validation/launches/h200-20260920-v1/test_h200.sbatch'
```


## Two-A40 corrected run (2026-09-21, passed)

Job **17533932** completed on **iris6**, exit 0, elapsed 22m10s: global batch 8,
UTD 20, FSDP 2, three updates and independent-process checkpoint restoration.
`EXPOLearner.update` now restores RNG placement to the learner mesh after
single-device inference. Both actual parameter partitioning and all learning /
restoration checks passed. See [VALIDATION.md](VALIDATION.md) for memory, timing
and compilation findings; steady-state update speed is not yet established.

The current `test_a40x2.sbatch` requests two A40s in the `iris` account/partition
and writes results under `/iris/u/kevinon/outputs/expo-ft-docker-validation/`.
It expects both `expo_ft.py` (the patched production learner module) and
`learner_smoke.py` beside the submission script. These are staged locally and
bound read-only over the unchanged image; library imports stay inside the SIF.
It enables JAX compile/cache-miss diagnostics and propagates phase failures.

Successful submission snapshot and rerun command:

```sh
ssh sc-codex 'sbatch --chdir=/iris/u/kevinon/workspace/expo-ft-docker-validation/launches/a40x2-rngfix-20260921-v1 /iris/u/kevinon/workspace/expo-ft-docker-validation/launches/a40x2-rngfix-20260921-v1/test_a40x2.sbatch'
```

Results: `/iris/u/kevinon/outputs/expo-ft-docker-validation/gpu-17533932/`.
H200 job 17533604 remained pending at the final check.


## Longer update-speed measurement (2026-09-21, passed)

`benchmark_a40x2.sbatch` keeps the successful A40 x2 / batch 8 / UTD 20 / FSDP 2
settings and runs **20 updates**, then checkpoint restoration. It requests 12
CPU threads, 80 GiB RAM and 2 hours. Prepare a submission snapshot containing
this script, `learner_smoke.py` and the patched production `expo_ft.py`, and pass
that directory as sbatch's `--chdir`, as for the three-update validation above.

Job **17539309** completed on iris6 in 39m11s. Updates 4–20 had no further
`_update_jit` XLA compilation and averaged **47.90 s/update**; the last ten
averaged **47.62 s/update**. All learning and checkpoint checks passed. See
[VALIDATION.md](VALIDATION.md) for measurement scope, memory and source snapshot.
To summarize the extracted job archive:

```sh
python docker/learner/summarize_speed.py /path/to/extracted/results
```

The analyzer writes `speed-summary.json`, preserving per-update times and the
post-compilation window rather than subtracting compilation from an early call.
