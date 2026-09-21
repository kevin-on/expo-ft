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
