# Checkpoint-owned model configuration

SFT and EXPO have different training configs. Checkpoints are the source of
model settings for evaluation and resume; hardware, ports and output directories
are separate. `checkpoint-config` was used as a reference, not merged.

## SFT

Use OpenPI's regular preset and CLI overrides. `scripts/train.py` stores a v2
`assets/config.json` alongside each step's normalization. It contains the CLI
and a JSON comparison document of the fully resolved TrainConfig (including
object types). Loading invokes the normal parser, then rejects changed settings.
The comparison document is not a Python object deserializer. Keep matching source;
this guards configuration defaults, not arbitrary changes in model implementation.
Programmatic training callers must pass `config_args` reproducing their config.

`omit_image_keys` controls both model token omission and the shared image mask.
Physical camera-eye selection and robot1 mirroring remain task/client concerns.

## New online run

```
python train_pi_robo.py --config=configs/model/expo_ft_pi_config.py \
  --initial_sft_checkpoint=/path/to/sft/4999 ...
```

Only the SFT step is needed: model architecture, Cartesian state/action format,
action horizon, image resize, camera omission, preprocessing and normalization
come from it. Removed `pi05_config_name`, `pi05_resize_size`, `pi05_omit_image_keys`,
`pi05_weight_loader_path`, `pi05_assets_dir`, `pi05_asset_id` are rejected rather
than silently competing with the saved config.

EXPO's config retains critic/edit architecture and online learning settings.
`actor_lr` controls the EXPO edit actor. The Pi optimizer is explicit:
`pi05_learning_rate=2.5e-5`, constant schedule, AdamW b1=.9, b2=.95, eps=1e-8,
weight_decay=1e-10, clip_gradient_norm=1, no EMA. These preserve the previous
online preset behavior instead of inheriting an arbitrary SFT run's schedule.
The SFT freeze filter (e.g. LoRA parameter selection) is retained, and
`freeze_pi05_encoder` remains an online control.

Learner and inference use the same SFT step and EXPO config. The split identity
now includes the full EXPO config except the relocatable SFT directory, so
settings like `use_pnorm` cannot silently differ between peers.

## Online checkpoints, resume and eval

Each online step saves `model_config/config.json` in the same Orbax operation as
`agent` and `params`. It records the effective EXPO config, SFT recipe, normalization
hash, EXPO factory defaults, model-related task settings, robot count and replan
length. Tuples survive JSON round trips. The training entrypoint additionally
records batch/UTD/offline ratio/update/seed settings.

Resume with the existing run directory and `--resume`; omit `--config` and its
overrides. Saved model and learning settings are restored before initialization.
GPU topology/ports/output paths are not taken from the checkpoint.
`--initial_sft_checkpoint` may relocate original SFT assets; recipe and norm
content must match. The original checkpoint weights are loaded by Orbax on resume.

Single-robot eval:

```
python eval_droid_policy.py --checkpoint_kind=sft --checkpoint_dir=/sft/4999 ...
python eval_droid_policy.py --checkpoint_kind=online --checkpoint_dir=/online/6400 ...
```

Both use `expo_ft.env.checkpoint_policy`; online evaluation takes no model config
or action-selection override. Its replan length comes from the saved record.
Expose one GPU. A task mismatch or absent/incompatible metadata fails before any
robot listener. Two-robot evaluation is a separate implementation stage.

## Old artifacts

Old metadata is never silently interpreted using today's defaults. Explicit
migration needs the original source/config provenance. Keep originals read-only
while validating a migrated copy. The GPU regression makes a named TEST fixture
from historical SFT CLI metadata; it does not migrate production artifacts.

## Verification

On a compute node in the matching environment, `JAX_PLATFORMS=cpu python
 tests/test_model_config.py` checks round trips, wrist modes, override retention,
relocation and mismatch rejection without devices. `tests/gpu/checkpoint_config.py`
uses an existing SFT checkpoint and recorded HDF5 only. Run `sft`, `train`, then
`restore` in separate processes: real inference, one small gradient update,
atomic model/config save, and action parity after independent restore.

For a v1 SFT artifact with recorded CLI/source revision, check a **copy** with:

```bash
python scripts/migrate_sft_config.py /copied/4999 --reference-openpi=/checkout/at-original-commit
# Inspect the verified JSON, then add --write to upgrade that copy (keeps config.v1.json).
```

Migration requires a clean original OpenPI checkout with the recorded commit and
compares its parser result with the current one. Artifacts without original
settings need a provenance audit, not a guessed preset. Old online checkpoints
without `model_config` remain on their original eval code until their complete
effective EXPO settings can be supplied and validated; this change does not
invent metadata for them or alter existing artifacts.

BC/RTC experimental trainers retain their legacy config presets; this checkpoint
contract currently covers SFT and `EXPOLearner` (sync, split and async). The checkpoint-driven eval path supports SFT/EXPO; explicitly selected BC/RTC
configs retain their existing evaluation path and are not covered by this contract.

ILIAD validation (2026-09-29, H200, no robot access): real mixed20 wrist-on SFT
inference passed; one batch-2/UTD-1 EXPO update with N=2 was saved, then restored
in a separate process with matching actions (rtol 1e-4, atol 1e-5), N and step.
These are correctness checks, not training throughput benchmarks.

Companion OpenPI revision: `590fa99` (branch `multi-robot` after integration).
Wrist-off real inference and the OpenPI tiny-model save/resume test also passed.
Full validation evidence is under
`/scr/kevinon/workspace/expo-ft-validation/20260929-model-config` and
`/iliad/u/kevinon/outputs/expo-ft/validation/20260929-model-config`.

The coordinated UI is `eval_sft_robots.py`; it uses these same model loaders and
takes no model config. See [two-robot eval](two_robot_eval.md).
