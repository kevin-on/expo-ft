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

    def compact(self, config, *, kind='sft', actor=None):
        from expo_ft.eval import checkpoint as ck
        import ml_dtypes
        base = {'llm': np.ones((2,), ml_dtypes.bfloat16), 'encoder': np.zeros((2,), np.float32)}
        cfg, _, _ = mc.load_sft(config)
        frozen, _ = ck.split_actor(base, cfg)
        trained = {'llm_lora': np.full((2,), 2, np.float32), 'encoder': np.full((2,), 3, np.float32)}
        meta = dict(kind=kind, training_run_id='tiny', checkpoint_path='sft/tiny/checkpoints/1',
                    hash_algorithm='xxh3_128', base_hash=ck.fingerprint(frozen),
                    files=mc.sft_files(config.initial_sft_checkpoint), replan_steps=8)
        with ck.pack({'actor': trained if actor is None else actor}, meta) as b:
            # Deliberately save ONLY the packet; metadata must not require adjacent assets.
            path = self.root/'standalone.bin'
            with b.open() as src, path.open('wb') as dst:
                dst.write(src.read())
        return path, base, trained

    def test_compact_config_norm_online_settings_and_relocation(self):
        for off in (False, True):
            with self.subTest(off=off):
                config, _ = self.fixture(off)
                config.pi05_learning_rate = 1e-5
                record = mc.make_record(config, task_config(), 8)
                path, _, _ = self.compact(config)
                restored = mc.restore_config(record, path, self.root/'base')
                (self.root/'base').mkdir(exist_ok=True)
                kwargs, cfg, _, _ = mc.build_online(restored)
                self.assertNotIn('initial_sft_base', kwargs)
                self.assertEqual(cfg.lr_schedule.peak_lr, 1e-5)
                self.assertEqual('left_wrist_0_rgb' in cfg.model.omit_image_keys, off)
                full, _, full_hash = mc.load_sft(config)
                compact, _, compact_hash = mc.load_sft(restored)
                self.assertEqual(full_hash, compact_hash)
                for key, stats in full.data.create(full.assets_dirs, full.model).norm_stats.items():
                    actual = compact.data.create(compact.assets_dirs, compact.model).norm_stats[key]
                    np.testing.assert_array_equal(stats.mean, actual.mean)
                # Full -> compact changes only relocatable paths, including old records
                # which predate the optional initial_sft_base field.
                record['config'].pop('initial_sft_base', None)
                expected = mc.make_record(restored, task_config(), 8)
                from types import SimpleNamespace
                mc.validate_agent(record, SimpleNamespace(actor=SimpleNamespace(checkpoint_record=expected)))

    def test_compact_requires_base_and_rejects_online_packet(self):
        config, _ = self.fixture()
        path, _, _ = self.compact(config)
        config.initial_sft_checkpoint = str(path)
        with self.assertRaisesRegex(ValueError, 'initial_sft_base'):
            mc.build_online(config)
        self.compact(config, kind='online')
        with self.assertRaisesRegex(ValueError, 'must be SFT'):
            mc.load_sft(config)

    def test_compact_loader_matches_full_and_rejects_bad_weights(self):
        from flax import nnx
        import jax
        import ml_dtypes
        from expo_ft.utils.train_utils import FlexibleCheckpointWeightLoader
        from expo_ft.eval import checkpoint as ck
        config, _ = self.fixture()
        path, base, trained = self.compact(config)
        config.initial_sft_checkpoint = str(path)
        config.initial_sft_base = str(self.root/'base'); Path(config.initial_sft_base).mkdir()
        _, cfg, _, _ = mc.build_online(config)
        full = {'llm': base['llm'], **trained}
        refs = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), full)
        class Tiny(nnx.Module):
            def __init__(self):
                for k, v in full.items(): setattr(self, k, nnx.Param(v))
        with patch('openpi.models.model.restore_params', return_value=base), \
                patch('flax.nnx.eval_shape', return_value=Tiny()), \
                patch('expo_ft.utils.train_utils._restore_pi05_params', return_value=full):
            loaded = cfg.weight_loader.load(refs)
            expected = FlexibleCheckpointWeightLoader('/unused').load(refs)
            for k in expected:
                np.testing.assert_array_equal(loaded[k], expected[k])
                self.assertEqual(loaded[k].dtype, expected[k].dtype)
            # Hash mismatch before model construction; original file is unchanged.
            wrong = {**base, 'llm': np.zeros(2, ml_dtypes.bfloat16)}
            with patch('openpi.models.model.restore_params', return_value=wrong):
                with self.assertRaisesRegex(ValueError, 'frozen base'):
                    cfg.weight_loader.load(refs)
            for actor, message in [({'encoder': trained['encoder']}, 'Missing'),
                                   ({**trained, 'extra': np.ones(1)}, 'extra'),
                                   ({**trained, 'encoder': np.ones(3)}, 'shape')]:
                with self.subTest(message=message):
                    self.compact(config, actor=actor)
                    with self.assertRaisesRegex(ValueError, message):
                        cfg.weight_loader.load(refs)
            # Config/norm text changing after build_online is rejected.
            self.compact(config)
            meta = ck.read_metadata(path)
            meta['files']['assets/test/data/norm_stats.json'] += ' '
            with ck.pack({'actor': trained}, meta) as b, b.open() as src:
                path.write_bytes(src.read())
            with self.assertRaisesRegex(ValueError, 'changed after'):
                cfg.weight_loader.load(refs)

    def test_online_export_after_compact_initialization(self):
        from expo_ft.eval import checkpoint as ck
        config, _ = self.fixture()
        path, base, trained = self.compact(config)
        config.initial_sft_checkpoint = str(path)
        record = mc.make_record(config, task_config(), 8)
        step = self.root/'online/1'; (step/'model_config').mkdir(parents=True)
        (step/'model_config/config.json').write_text(json.dumps(record))
        params = dict(actor_params={**base, **trained}, batch_encoder_params={}, edit_actor_params={})
        state = dict(actor_train_state={'params': {}}, target_critic={'params': {}})
        with patch.object(ck, 'restore_tree', side_effect=[params, state]), \
                ck.export(step, 'online', checkpoint_path='online/test/checkpoints/1', training_run_id='test') as b:
            meta = ck.manifest(b)['metadata']
            self.assertEqual(meta['kind'], 'online')
            self.assertEqual(meta['files']['assets/config.json'], mc.sft_files(path)['assets/config.json'])

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
