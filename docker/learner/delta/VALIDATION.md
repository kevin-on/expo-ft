# Delta H200 validation — 2026-09-21

The full robot-free validation completed on Delta node `gpue04`, allocation
`22287469`: one NVIDIA H200 (143771 MiB), 12 CPU cores, 250G host RAM;
Apptainer 1.5.1, NVIDIA driver 595.71.05, JAX 0.5.3. The existing allocation
remained running after these tests.

| Setting / result | Small batch | Example default batch |
| --- | ---: | ---: |
| Global batch size | 8 | 64 |
| UTD / FSDP devices | 20 / 1 | 20 / 1 |
| Updates completed | 20 | 20 |
| Post-compilation updates measured | 17 | 17 |
| Mean update duration | 3.590 s | 23.125 s |
| Min–max update duration | 3.579–3.603 s | 23.095–23.147 s |
| JAX peak live allocation, train | 45.85 GiB | 78.12 GiB |
| Checkpoint save and independent restore | Passed | Passed |

Each case passed finite inference/loss checks, parameter-change checks for actor
LoRA, critic, edit actor and image encoder, and checkpoint hash/step/action
reproduction. `_update_jit` compiled on updates 1–3; updates 4–20 provide the
reported means. Timing includes `update()` and synchronization plus inference
cache refresh, excluding batch preparation and parameter hash checks. No action
inference is interleaved between those updates. Separate warmed inference calls
were approximately 60–69 ms for a 16×7 action chunk.

Sampled `nvidia-smi` memory peaked at 131801 MiB across the run. This includes
reserved allocator memory and is different from JAX live allocation; the pool
limit was 0.95, with preallocation disabled.

The input was `pi05_base` plus initialized LoRA and one 226-transition recording.
Normalization statistics were generated as test fixtures. No robot connection,
policy-quality evaluation or network latency test was performed.

## Reproduction evidence

- Full results, phase logs and retained checkpoints:
  `/work/hdd/bgqe/kon/expo-ft/validation/h200-22287469/`.
- Reusable inputs: `/projects/bgqe/kon/expo-ft/baseline-20260921/`.
- SIF SHA256:
  `3ca61c8fb26fab7597b01c22ca1e5599657a59ae155117010e0185ac11f2f7ef`.
- OpenPI commit: `46407a41183b037313a383ff679683f2773b766d`.
- Learner SHA256 (`expo_ft/agents/alg/expo_ft.py`):
  `d37c722a4e65c7741d8d11e386eb549bbdbff410d11b2610bf54491175e63db7`.
- Harness SHA256 (`tests/gpu/learner_smoke.py`):
  `18c6fdcf255e747d2d5f9f7f84b9eab4570bc9f8cd9663f043c7121bd1ac281e`.

That full run used the archived, experiment-specific launcher, before the reusable
Delta wrapper was added. Its runtime source was compared with branch
`codex/multi-robot-sync` at `2d4b455`: 57 selected learner/config/root/test files and all 95 OpenPI source files
matched. The only differing selected file was the robot-1 camera JSON,
which the robot-free harness does not use. The image contains the older source;
the experiment overlaid the corrected learner and harness. The new `run.sh`
instead snapshots the current tracked learner source so that future code changes
are not silently hidden by the image's older copy.

## Reusable launcher integration check

The new Delta launcher also passed a fresh run in the same existing H200
allocation: **batch 8, UTD 20, FSDP 1, one update**, followed by checkpoint save
and independent-process restore with matching parameters, step and action.
Both phase exit codes and the launcher exit code were zero; `summary.json`
reported success. Preflight confirmed that EXPO and `train_pi_robo` loaded from
`/source`, while OpenPI loaded from the image. All 21 weight/tokenizer files
passed checksum verification.

Evidence: `/work/hdd/bgqe/kon/expo-ft/validation/delta-launcher-23ecf87-j22287469/`.
The source manifest records the candidate commit before this result paragraph
was added; the launcher code is unchanged. The integration check deliberately
has no steady-state speed estimate. The 20-update measurements above remain the
performance evidence. No new allocation or robot connection was made.
