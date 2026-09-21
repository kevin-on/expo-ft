# Robot-free A40 validation — 2026-09-20

The Docker-derived learner/inference environment passed repeated EXPO updates
and checkpoint restoration on one NVIDIA A40, with a reduced validation batch.
The example training defaults did not fit in the tested configuration.

## Reproducible environment

- Repository HEAD: `f92c32a0d1eae3aca2282ff1787228c60b55a9be`.
- OpenPI: `46407a41183b037313a383ff679683f2773b766d`.
- Runtime: Apptainer 1.5.3, Docker base `python:3.11-slim-trixie`, Python 3.11.16,
  JAX/CUDA plugin 0.5.3, Flax 0.10.2, Optax 0.2.4. Exact packages are in
  `requirements-tested.txt`. CPU torch is an import/transform dependency;
  learner and inference use JAX CUDA.
- SIF: `/iris/u/kevinon/artifacts/expo-ft/expo-ft-learner-jax053.sif`.
- SHA256: `3ca61c8fb26fab7597b01c22ca1e5599657a59ae155117010e0185ac11f2f7ef`.
- GPU: one A40 on iliad5, driver 610.43.02. Job **17531495**.
- JAX preallocation disabled; `XLA_PYTHON_CLIENT_MEM_FRACTION=0.98`.
- The SIF, weights, recording, caches and detailed logs ran from job-local disk.
  Shared Python libraries were not mounted over the container environment.
- Base Pi0.5 weights plus initialized LoRA; all 21 weight/tokenizer objects were
  verified against GCS CRC32C and a SHA256 manifest (12,445,985,954 bytes total).
- One 226-transition teleoperation recording; statistics were computed from
  that recording solely as validation fixtures.

## Results

| Check | Outcome |
| --- | --- |
| Dependency consistency and learner import | Passed (134 packages checked) |
| CUDA initialization and a GPU matrix operation | Passed |
| Inference, shape 16x7 and finite actions | Passed |
| Batch 2 / UTD 1, three updates | Passed; finite metrics and changed actor, critic, edit actor, encoder parameters |
| Save and restore in a separate process | Passed; parameter hashes, step and same-RNG actions matched |
| Batch 64 / UTD 20 | Failed at first-update compilation/autotuning: GPU OOM allocating 1.89 GiB |
| Batch 8 / UTD 1, 95% pool | First update passed; second update OOM (job 17530754) |
| Batch 1 | Unsupported by current sampling path: squeeze removes singleton batch axis (job 17530842) |

The successful small-batch run's peak live JAX allocation was **42.879 GiB**.
The complete job's maximum nvidia-smi usage was **44,953 MiB (43.899 GiB)**,
including allocator pools and CUDA overhead. This leaves little headroom.
The failed default profile's recorded events do not capture a successful peak
for the full update and must not be treated as its required memory estimate.

Warm inference took **0.202–0.221 seconds** per action chunk in the successful
small-batch run, with 66.85 seconds on the first compilation call. Updates took
305.47, 162.36 and 159.45 seconds respectively. These are observed call times,
including any compilation; they are not isolated kernel benchmarks. Updating
works in this reduced configuration, but online-training throughput has not been
established. Network, robot latency and simultaneous clients were not tested.

Slurm reported COMPLETED for historical job 17531495 because the then-current
script tolerated failure of the optional default profile. Its run-status records
small-train=0, small-restore=0, default-train=1. The current script propagates a
requested default-profile failure to Slurm. To rerun only the successful profile:

```sh
ssh sc-codex 'sbatch --exclude=iliad6 --export=ALL,EXPO_VALIDATE_DEFAULTS=0 /iliad/u/kevinon/workspace/expo-ft-docker-validation/test_a40.sbatch'
```

## Evidence and limitations

Remote archive and summary:
`/iliad/u/kevinon/outputs/expo-ft-docker-validation/gpu-17531495/`.
Local extracted evidence:
`/scr/kevinon/tmp/expo-ft-gpu-validation/gpu-17531495/`.
The archive contains JSONL stage events, finite loss/parameter-change checks,
restore comparisons, per-second GPU samples and process logs. The large test
checkpoint was intentionally left in job-local scratch; it is not a retained
trained policy artifact. The public initial weights remain on shared storage.

The recording and initialized LoRA do not demonstrate task success. During this
single-GPU run, production algorithm files, workstation client environments and
robot hardware settings were not modified. The singleton-batch issue was recorded, not fixed. Neither
H200 nor L40S learner validation completed: iris9's L40S hit ECC errors before
model loading; H200 has not been used for this validation.


## Two-A40 validation after RNG placement fix — 2026-09-21

Job **17533932** on **iris6** completed with exit **0**, elapsed **22m10s**.
Configuration: two A40 GPUs, global batch **8**, UTD **20**, FSDP **2**, three
updates, 95% JAX pool limit with preallocation disabled.

The earlier two-GPU job 17533545 failed because inference left the learner RNG
on GPU 0 while the update state spanned both GPUs. `EXPOLearner.update` now
places both input learners' RNGs on their actor's replicated learner sharding
before entering JIT. It preserves RNG values and the inference-cache behavior.
The unchanged SIF ran with this production module and the harness copied to
node-local storage and bound read-only; their hashes are in `source-sha256.txt`.

- Two-GPU reduction and actual actor parameter partitioning passed: 37 of 71
  arrays were partitioned, representing 7,723,401,216 global bytes.
- All three updates passed finite-metric and parameter-change checks for actor
  LoRA, critic, edit actor and encoder.
- Checkpoint save and independent-process restore passed; parameter hashes,
  step count and same-RNG inference matched.
- Peak live JAX allocation: **30.83 GiB / 19.64 GiB**, GPU 0 / GPU 1.
  nvidia-smi peaks including allocator pools: **33411 / 33401 MiB**.
- Warm inference: **0.20–0.22 seconds** per action chunk.

Update call times were **403.11 / 269.91 / 271.15 seconds**. Compile diagnostics
show XLA compilation took **340.84 / 215.13 / 215.66 seconds**, respectively.
The second call changed state sharding and scalar weak types; the third changed
temperature optimizer moment weak types. These three calls do not establish
steady-state training speed or prove that every subsequent call recompiles.
The remainder of each call includes tracing, dispatch and execution, and must
not be reported as isolated GPU execution time.

Evidence: `/iris/u/kevinon/outputs/expo-ft-docker-validation/gpu-17533932/`.
Local extracted evidence:
`/scr/kevinon/tmp/expo-ft-gpu-validation/gpu-17533932/`.
Submission snapshot:
`/iris/u/kevinon/workspace/expo-ft-docker-validation/launches/a40x2-rngfix-20260921-v1/`.
The archive excludes the large job-local test checkpoint. This is robot-free
execution validation, not a full online-training or policy-quality test.
H200 job 17533604 remained pending at the final status check.


## Twenty-update A40 benchmark — 2026-09-21

Job **17539309** on **iris6** completed with exit **0**, elapsed **39m11s**.
It reused the same patched production learner, harness, SIF, recording and
model settings: A40 x2, global batch 8, UTD 20, FSDP 2, 95% allocator limit.
All 20 updates passed finite-metric and trainable-parameter-change checks.
Checkpoint save and independent-process restore passed, including same-RNG
inference and parameter/step comparisons.

| Update window | Mean call time | Range | XLA compilation of `_update_jit` |
| --- | --- | --- | --- |
| 1–3 | 412.27 / 277.71 / 279.31 s individually | — | 336.65 / 212.44 / 212.32 s respectively |
| 4–20 (17 calls) | **47.90 s** | 47.23–48.83 s | None observed |
| 11–20 (last 10 calls) | **47.62 s** | 47.23–48.02 s | None observed |

This establishes approximately **1.25 update calls/minute** after compilation
for this configuration. Each call includes UTD 20 and inference-cache refresh;
it is not one critic minibatch. The timer waits for JAX work to finish, excludes
batch preparation and parameter hash checks, and includes host dispatch as well
as device work. No inference runs between these repeated updates. Full online
robot collection/network overhead is outside the measurement.

Peak live JAX allocation was **31.82 / 19.64 GiB** per GPU; nvidia-smi peaks
including pools/overhead were **33375 / 33365 MiB**. Slurm batch MaxRSS was
34394133 KiB (about 32.80 GiB). The job requested 12 CPU threads and was adjusted
while pending from 96 GiB RAM to **80 GiB**, retaining headroom above the previous
run's roughly 49 GiB peak. Wall limit was 2 hours; training requested 20 updates.

Script: `benchmark_a40x2.sbatch`. Analyzer: `summarize_speed.py`, which correlates
update boundaries with compiler logs and deduplicates duplicate log handlers.
Submission snapshot:
`/iris/u/kevinon/workspace/expo-ft-docker-validation/launches/a40x2-speed-20260921-v1/`.
Its `submission.txt` records the Slurm RAM override to the original script.
Evidence: `/iris/u/kevinon/outputs/expo-ft-docker-validation/gpu-17539309/`.
Local evidence and computed `speed-summary.json`:
`/scr/kevinon/tmp/expo-ft-gpu-validation/gpu-17539309/`.
