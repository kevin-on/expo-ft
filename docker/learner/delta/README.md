# Delta H200 learner / inference runtime

Run the checked-out EXPO learner source using the validated x86-64 Apptainer
image. Libraries and pinned OpenPI stay inside the image; the launcher snapshots
tracked learner/config/harness files from the current working tree and places
that snapshot first on `PYTHONPATH`. No Python environment is read from NFS.
DeltaAI GH200 is ARM and needs a separate image.

`validate` runs the real replay loader, learner updates, inference, checkpoint
save, and an independent-process restore without connecting to a robot. It uses
`pi05_base` with initialized LoRA and normalization fixtures computed from the
recording. This verifies execution, memory and numerical checks, not policy
quality. See [measured H200 results](VALIDATION.md).

## Persistent inputs and results

Prepare an input directory with the following layout (large artifacts are not
committed). The SIF build recipe is in [the parent directory](../README.md).

```text
INPUT_ROOT/
  expo-ft-learner-jax053.sif
  openpi-cache/
    manifest.json
    openpi-assets/checkpoints/pi05_base/params/...
    ... tokenizer and other downloaded OpenPI assets ...
  demo/
    0/traj.hdf5
```

The cache manifest is the per-file SHA256/GCS CRC32C manifest accepted by
[`verify_assets.py`](../verify_assets.py). A recording's episode-parent directory
is required, not a single HDF5 filename. Assets are checked before model loading.
As of 2026-09-21, the prepared input root is
`/projects/bgqe/kon/expo-ft/baseline-20260921`.

Use `/projects/bgqe` for reusable inputs and `/work/hdd/bgqe` or
`/work/nvme/bgqe` for experiment results. These project filesystems survive a
compute allocation ending; this is not a backup guarantee. The launcher stages
the SIF, weights, recording, source and runtime caches under node-local `$TMPDIR`
(default `/tmp`). Allow at least 30 GiB free there for these inputs and compilation
caches, with more space for larger recordings. Checkpoints and logs are written
to the persistent output directory; the default two-case test needs about 33 GiB
for checkpoints alone. Keep outputs outside the repository.

Every invocation requires a **new output directory** and refuses to overwrite
one. Results include `exit-code.txt`, `summary.json`, phase logs, JSONL events,
checkpoints, source snapshots/hashes, image hash, dependency versions and sampled
GPU memory. Scratch is intentionally retained, with its path in `local-stage.txt`;
it is temporary and should be cleaned up once no process uses it. JAX cache data
stays on the compute node, not in the otherwise persistent `jax-cache/` mountpoint.

## In an existing interactive allocation

In the allocated compute shell, from the repository root:

```bash
export EXPO_INPUT_ROOT=/projects/bgqe/kon/expo-ft/baseline-20260921
export EXPO_OUTPUT_DIR=/work/hdd/bgqe/kon/expo-ft/validation/h200-$(date +%Y%m%d-%H%M%S)
export EXPO_IMAGE_SHA256=3ca61c8fb26fab7597b01c22ca1e5599657a59ae155117010e0185ac11f2f7ef
bash docker/learner/delta/run.sh validate
```

Do not nest `srun` inside an `srun --pty bash` shell. For a persistent interactive
shell, run the allocation from login-node tmux and reattach on that same login
node. The launcher does not request or release an allocation.

Defaults are batch sizes **8 and 64**, **20 updates each**, **UTD 20**, **FSDP 1**,
and a 100-minute timeout for each train/restore phase. For a shorter integration
check, set `EXPO_BATCH_SIZES=8 EXPO_UPDATES=1`; that does not measure steady speed.
Other overrides: `EXPO_UTD`, `EXPO_FSDP_DEVICES`, `EXPO_PHASE_TIMEOUT`,
`EXPO_MEMORY_FRACTION` (default `0.95`, with preallocation disabled), `EXPO_IMAGE`,
and `EXPO_DATASET`. GPU count/CPU/RAM belong to the Slurm request, not these
settings. The default batch-64 test was validated on one H200, not one A40.

`run.sh preflight` stages and checks the assets and performs a small CUDA/JAX
matrix multiplication only. It also requires a new output directory.

## Submit a new batch job

From the repository root on a Delta login node:

```bash
mkdir -p logs
export EXPO_INPUT_ROOT=/projects/bgqe/kon/expo-ft/baseline-20260921
export EXPO_OUTPUT_ROOT=/work/hdd/bgqe/kon/expo-ft/validation
export EXPO_IMAGE_SHA256=3ca61c8fb26fab7597b01c22ca1e5599657a59ae155117010e0185ac11f2f7ef
sbatch docker/learner/delta/submit.sbatch
```

The wrapper requests one H200, 12 CPU cores, 250G RAM and two hours on
`gpuH200x8` with account `bgqe-delta-gpu`. Recheck `accounts` and current partition
policy. Resource/time flags can be overridden **before** the script filename,
e.g. `sbatch --time=48:00:00 docker/learner/delta/submit.sbatch`.
48 hours is the normal partition limit; idle allocations also consume quota.
With the billing weights checked on 2026-09-21, 12 cores plus 250G stay within
one H200's charge, and a 48-hour allocation costs 144 allocation hours.
See [Delta job accounting](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html).

Save the returned job ID. Use `squeue -j JOBID`, `scontrol show job JOBID`, and
`tail -f logs/expo-h200-validation-JOBID.out`. The result directory is
`$EXPO_OUTPUT_ROOT/expo-h200-validation-JOBID`. Cancel only the intended job with
`scancel JOBID`. The batch wrapper launches one `srun` step; the interactive
instructions above run directly in the already allocated shell.

## Other learner / inference commands

The launcher also accepts an explicit container command, for example:

```bash
bash docker/learner/delta/run.sh python train_pi_robo.py --help
```

The container working directory is the source snapshot. Paths inside it are
`/openpi-cache`, `/demo`, `/output`, `/cache`, and `/source`. Supply model/config,
dataset, output and network arguments for the specific experiment. Only tracked
learner/config/root/harness files are snapshotted; new untracked source must be
added to Git first. Client hardware code is deliberately excluded. OpenPI and
third-party dependency changes require rebuilding the image; the source overlay
does not install dependencies.

WandB defaults to disabled. For online experiments, supply `EXPO_WANDB_MODE=online`,
`WANDB_API_KEY` and `WANDB_RUN_GROUP` through the shell/secret configuration and
pass `--project_name=mtexpo --run_name=<unique-name>` to `train_pi_robo.py`.
Never put credentials in scripts or commits. This launcher does not configure
workstation-to-Delta tunnels, real-robot normalization, or robot rollout clients;
the robot-free result does not validate that network/control path.
