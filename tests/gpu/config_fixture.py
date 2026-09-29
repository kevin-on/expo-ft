"""Explicit synthetic config metadata for benchmarks using bare params/assets.

Not a production checkpoint migration. Tests declare the historical preset as part
of their fixture, just as they previously did through the runtime config.
"""
import json
from pathlib import Path
import shutil


def configure(config, root, params, assets, asset_id):
    from openpi.training import config as op, checkpoint_config
    root=Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not (root/'params').exists():
        (root/'params').symlink_to(Path(params).resolve(), target_is_directory=True)
    destination=root/'assets'/asset_id
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(assets)/asset_id/'norm_stats.json', destination/'norm_stats.json')
    args=['expo_pi05_droid_lora_finetune_sft_cartesian_state', '--exp-name=validation',
          '--data.repo-id='+asset_id, '--data.assets.asset-id='+asset_id]
    record=checkpoint_config.make_record(op.cli(args),args)
    (root/'assets/config.json').write_text(json.dumps(record))
    config.initial_sft_checkpoint=str(root)
