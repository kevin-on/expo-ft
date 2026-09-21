# Four-chip TPU learner verification

The TPUQ entrypoint is `profiling/run_tpu_profile.sh`: pool `usc2`, `v4-8`, all
four physical chips, one JAX process, FSDP 4, global batch 4, UTD 1. Each of
baseline, baseline-repeat, and fixed runs six actual learner updates. Batch 4
is required by the harness's device divisibility check; the A40 run used batch 2.
The model remains N=8, edit samples=8, Q ensemble=10, with trainable critic encoder.

Baseline is commit `f92c32a0d1eae3aca2282ff1787228c60b55a9be`. Fixed includes the
profiling worktree's candidate changes captured when this branch was created.
The candidate's single-device placement normalization does not run under FSDP 4.
Numerical tolerance stays rtol=1e-4, atol=1e-5. The baseline repeat must also pass.
Both finite-learning success and strict comparison results should be reported;
a numerical comparison failure does not by itself mean the learner cannot train.

Inputs live in `gs://soe-iris-kevin-usc2/expo-ft/assets/validation-20260921.tar`,
in US-CENTRAL2, matching the TPU pool. This packages the original GPU-validation
OpenPI cache/manifest and recorded demo. Workers stage it to local disk and
verify every checkpoint SHA256 plus the demo checksum before loading. The
checkpoint is pi05_base plus freshly initialized LoRA, not a task-trained policy.
The runtime uses the repository's pinned TPU requirements, JAX 0.5.3 and CPU
PyTorch 2.7.1, in a dedicated worker-local uv environment; OpenPI is pinned to
46407a41183b037313a383ff679683f2773b766d. It does not use the CUDA SIF image.

Results are uploaded after every variant and on exit to
`gs://soe-iris-kevin-usc2/expo-ft/learner-profile/<task>/attempt-<run>/`.
Outputs include inputs' hashes, compile logs, update timings, per-device memory
stats where available, parameter partitioning, finite metrics, changed parameter
groups, final trainable arrays/actions and comparison.json. Device work is
synchronized for timing. Batch processing and parameter hashing are outside the
timed update region. Baseline-repeat reuses baseline's compilation cache; fixed
uses a separate cold cache. No compilation cache is uploaded.

This profile uses --skip-checkpoint, matching the GPU profile: it does not test
checkpoint save/restore or robot task success. Spot loss starts a fresh six-update
experiment in a new attempt directory. Existing completed-attempt results remain
in GCS. No large artifacts need to be downloaded out of GCP for routine review.

Submit the pushed commit from mini's existing TPUQ installation:

```bash
.venv/bin/tpuq submit /path/to/run_tpu_profile.sh \
  --repo git@github.com:kevin-on/expo-ft.git \
  --commit FULL_40_CHARACTER_SHA --name expo-v4-fsdp4-learner-profile
```

## First TPU run and common compatibility repair

Job `3d1e291267df` (revision `5ebf303`) loaded the model and completed inference
on v4-8, with 37 partitioned actor arrays. The first update failed because
`sample_actions()` left `agent.rng` committed to TPU 0 while critic state used
all four devices. This was a device-set mismatch, not OOM.

Subsequent runs apply `place_update_rng` to both baseline and fixed before their
update implementations. It replicates only the RNG onto the training mesh,
preserving its value and all parameter/optimizer shardings. Baseline therefore
means f92c32a plus this disclosed common compatibility repair. The snapshot
records this in compatibility-patches.txt. Numerical tolerances are unchanged.

## Completed v4-8 result (2026-09-21)

Job `61fe3612e72b`, task `767d9294440548f2`, ran revision
`9902cd32ac967c34c32cded79525dcd31c0f2a22`. All three variants passed six real
updates and post-update inference (18 total updates). Inputs, all recorded
metrics, final trainable arrays and final actions matched exactly between
baseline and fixed; the baseline-repeat control passed too.

| Variant | Update executable counts | Warm update mean | Maximum per-chip memory |
| --- | --- | --- | --- |
| Baseline + RNG repair | 1,2,3,3,3,3 | 1.805 s (updates 4–6) | 19.399 GiB |
| Baseline repeat | 1,2,3,3,3,3 | 1.829 s (updates 4–6) | 19.399 GiB |
| Candidate + RNG repair | 1,2,2,2,2,2 | 1.886 s (updates 3–6) | 19.399 GiB |

The scheduler status is **FAILED** only because the candidate requires two
executables instead of the required one. The numerical comparison passed with
zero observed differences. This validates robot-free FSDP-4 learning at batch 4,
UTD 1; it does not validate the no-recompile optimization, larger batches,
checkpoint save/restore, or robot task performance. The small warm timing sample
is not sufficient to conclude a steady-state speed difference.

The candidate's first update changes critic/optimizer array shardings, which is
visible in its next recorded input signature; single-device normalization does
not handle this four-device case. No additional FSDP placement optimization was
applied. 37 actor parameter arrays were partitioned across the four devices.

See `tpu-v4-results.json` for precise timings, memory and equality diagnostics.
The original GCS comparison's fixed warm-mean field incorrectly includes update
2's compilation; the summary uses the actual stable-cache suffix. The reporting
code now calculates this suffix for either variant, without changing pass/fail.
Inputs and large outputs remain in the matching US-CENTRAL2 bucket.
