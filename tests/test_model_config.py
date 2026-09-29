"""Checkpoint configuration regressions; no cameras, sockets, model initialization or GPU."""
import copy
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from configs.model.expo_ft_pi_config import get_config
from configs.task.pick import get_config as task_config
from openpi.training import config as op_config, checkpoint_config as recipe
from openpi.shared import normalize
from expo_ft.utils import model_config as mc

PRESET = 'expo_pi05_droid_lora_finetune_sft_cartesian_state'


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def fixture(self, off=False):
        root = self.root / ('off' if off else 'on')
        (root/'params').mkdir(parents=True)
        args = [PRESET, '--exp-name', 'test', '--data.repo-id', 'test/data',
                '--data.assets.asset-id', 'test/data', '--model.omit-image-keys']
        args += ['left_wrist_0_rgb', 'right_wrist_0_rgb'] if off else ['right_wrist_0_rgb']
        cfg = op_config.cli(args)
        rec = recipe.make_record(cfg, args)
        stats = {k: normalize.NormStats(mean=np.zeros(7), std=np.ones(7), q01=-np.ones(7), q99=np.ones(7))
                 for k in ('state', 'actions')}
        normalize.save(root/'assets/test/data', stats)
        (root/'assets/config.json').write_text(json.dumps(rec))
        config = get_config(); config.initial_sft_checkpoint = str(root)
        return config, rec

    def test_wrist_on_off_and_online_optimizer(self):
        for off in (False, True):
            with self.subTest(off=off):
                config, _ = self.fixture(off)
                config.N = 4
                config.pi05_learning_rate = 1e-5
                kwargs, cfg, resize, cls = mc.build_online(config)
                self.assertEqual('left_wrist_0_rgb' in cfg.model.omit_image_keys, off)
                self.assertEqual(kwargs['N'], 4)
                self.assertEqual(cfg.lr_schedule.peak_lr, 1e-5)
                self.assertEqual(cfg.data.assets.asset_id, 'test/data')
                self.assertEqual(resize, 224)
                data = cfg.data.create(cfg.assets_dirs, cfg.model)
                transformed = data.model_transforms.inputs[0]({'image': {k: np.ones((2,2,3)) for k in
                    ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},
                    'image_mask': {k: True for k in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')}})
                self.assertEqual(bool(transformed['image_mask']['left_wrist_0_rgb']), not off)

    def test_preset_drift_and_unrecorded_override_rejected(self):
        config, record = self.fixture()
        cfg = recipe.restore(record)
        with patch.object(op_config, 'cli', return_value=dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, action_horizon=32))):
            with self.assertRaisesRegex(ValueError, 'changed'):
                recipe.restore(record)
        with self.assertRaisesRegex(ValueError, 'CLI'):
            recipe.make_record(dataclasses.replace(cfg, batch_size=999), record['config_args'])

    def test_online_roundtrip_overrides_and_relocation(self):
        config, _ = self.fixture()
        config.N = 4; config.n_edit_samples = 4; config.use_pnorm = True
        record = mc.make_record(config, task_config(), 8)
        restored = mc.restore_config(json.loads(json.dumps(record)))
        self.assertEqual(restored.to_dict(), config.to_dict())
        other = self.root/'relocated'
        import shutil
        shutil.copytree(config.initial_sft_checkpoint, other)
        self.assertEqual(mc.restore_config(record, other).N, 4)
        stats = other/'assets/test/data/norm_stats.json'
        stats.write_text(stats.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'normalization'):
            mc.restore_config(record, other)

    def test_task_horizon_and_duplicate_config_rejected(self):
        config, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, 'horizon'):
            mc.make_record(config, task_config(), 100)
        record = mc.make_record(config, task_config(), 8)
        with self.assertRaisesRegex(ValueError, 'replan'):
            mc.check_task(record, task_config(), 4)
        task = task_config(); task.language_instruction = 'different'
        with self.assertRaisesRegex(ValueError, 'language'):
            mc.check_task(record, task, 8)
        config.pi05_omit_image_keys = ()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            mc.build_online(config)

    def test_resume_restores_effective_flags(self):
        from absl import flags
        from ml_collections import config_flags
        config, _ = self.fixture()
        values = flags.FlagValues()
        config_flags.DEFINE_config_dict('config', config, flag_values=values, lock_config=False)
        config_flags.DEFINE_config_dict('config_task', task_config(), flag_values=values, lock_config=False)
        for key,value in dict(output_dir=str(self.root),run_name='run',initial_sft_checkpoint='',update_type='episode').items():
            flags.DEFINE_string(key,value,'',flag_values=values)
        flags.DEFINE_bool('resume',False,'',flag_values=values)
        for key,value in dict(replan_steps=8,num_robot=2,batch_size=64,utd_ratio=20,num_updates=3,step_interval=50,seed=42).items():
            flags.DEFINE_integer(key,value,'',flag_values=values)
        flags.DEFINE_float('offline_ratio',0.,'',flag_values=values)
        values(['test'])
        record=mc.configure_training(values)
        step=self.root/'run/checkpoints/10';(step/mc.CONFIG_ITEM).mkdir(parents=True)
        (step/'_CHECKPOINT_METADATA').write_text('{}')
        (step/mc.CONFIG_ITEM/mc.CONFIG_FILE).write_text(json.dumps(record))
        values.resume=True; values.batch_size=1; values.config.N=99
        restored=mc.configure_training(values)
        self.assertEqual(values.batch_size,64)
        self.assertEqual(values.config.N,8)
        self.assertEqual(restored,record)
        values['config'].present=1
        with self.assertRaisesRegex(ValueError,'remove --config'):
            mc.configure_training(values)

    def test_save_and_missing_metadata(self):
        config, _ = self.fixture()
        record = mc.make_record(config, task_config(), 8)
        path = self.root/'10'/mc.CONFIG_ITEM/mc.CONFIG_FILE
        path.parent.mkdir(parents=True);path.write_text(json.dumps(record))
        self.assertEqual(mc.read_record(path.parent.parent), record)
        changed = copy.deepcopy(record); changed['factory_defaults']['N'] = 99
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'defaults changed'):
            mc.read_record(path.parent.parent)
        with self.assertRaisesRegex(ValueError, 'Missing online config'):
            mc.read_record(self.root/'999')
        with self.assertRaisesRegex(ValueError, 'migrate'):
            recipe.restore({'version': 1, 'config_args': ['old']})


if __name__ == '__main__':
    unittest.main()
