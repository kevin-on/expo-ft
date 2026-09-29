"""Checkpoint-owned SFT and online settings. No device or robot initialization."""
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path

from ml_collections import ConfigDict
from openpi.training import checkpoint_config as sft_config
from openpi.training import optimizer

CONFIG_ITEM = 'model_config'
CONFIG_FILE = 'config.json'
def pack(value):
    if isinstance(value, tuple):
        return {'__tuple__': [pack(v) for v in value]}
    if isinstance(value, list):
        return [pack(v) for v in value]
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Online config must contain plain values, got {type(value)}')


def unpack(value):
    if isinstance(value, dict):
        if set(value) == {'__tuple__'}:
            return tuple(unpack(v) for v in value['__tuple__'])
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v) for v in value]
    return value


LEGACY_FIELDS = ('pi05_config_name', 'pi05_resize_size', 'pi05_omit_image_keys',
                 'pi05_weight_loader_path', 'pi05_assets_dir', 'pi05_asset_id')


def load_sft(config):
    legacy = set(config).intersection(LEGACY_FIELDS)
    if legacy:
        raise ValueError(f'Remove duplicate SFT settings {sorted(legacy)}; use initial_sft_checkpoint')
    root = Path(config.initial_sft_checkpoint).resolve()
    recipe = sft_config.read_record(root)
    cfg = sft_config.restore(recipe)
    if not getattr(cfg.data, 'use_cartesian_state', False) or cfg.data.output_action_dim != 7:
        raise ValueError('EXPO DROID requires Cartesian state and 7D velocity actions')
    asset_id = cfg.data.assets.asset_id or cfg.data.repo_id
    stats = root / 'assets' / asset_id / 'norm_stats.json'
    if not (root / 'params').is_dir() or not stats.is_file():
        raise ValueError('SFT checkpoint must contain params and its declared normalization asset')
    cfg = sft_config.with_assets(cfg, root)
    return cfg, recipe, hashlib.sha256(stats.read_bytes()).hexdigest()


def build_online(config):
    cfg, recipe, norm_hash = load_sft(config)
    kwargs = dict(config)
    root = Path(kwargs.pop('initial_sft_checkpoint')).resolve()
    cls = kwargs.pop('model_cls')
    # Explicit online Pi optimizer; SFT run length/schedule/EMA do not leak into online training.
    lr = kwargs.pop('pi05_learning_rate')
    adam = optimizer.AdamW(**{k: kwargs.pop('pi05_' + field) for k, field in
        [('b1', 'adam_b1'), ('b2', 'adam_b2'), ('eps', 'adam_eps'),
         ('weight_decay', 'weight_decay'), ('clip_gradient_norm', 'clip_gradient_norm')]})
    from expo_ft.utils.train_utils import FlexibleCheckpointWeightLoader
    cfg = dataclasses.replace(cfg, weight_loader=FlexibleCheckpointWeightLoader(str(root/'params')),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=0, peak_lr=lr,
                                                 decay_steps=100_000, decay_lr=lr),
        optimizer=adam, ema_decay=None)
    return kwargs, cfg, cfg.data.model_image_resize, cls


def defaults_signature():
    from expo_ft.agents.alg.expo_ft import EXPOLearner
    return {k: sft_config.describe(v.default) for k, v in inspect.signature(EXPOLearner.create).parameters.items()
            if v.default is not inspect.Parameter.empty}


def make_record(config, task, replan_steps, num_robot=2):
    cfg, recipe, norm_hash = load_sft(config)
    if not 0 < replan_steps <= cfg.model.action_horizon:
        raise ValueError('replan_steps must fit the SFT action horizon')
    if (task.action_space, task.gripper_action_space) != ('cartesian_velocity', 'velocity'):
        raise ValueError('SFT checkpoint requires Cartesian/gripper velocity task')
    values = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
    # Resolve tuples and configuration references before persisting.
    values = pack(values)
    return dict(version=1, config=values, sft=recipe, norm_sha256=norm_hash,
        factory_defaults=defaults_signature(),
        task=dict(language_instruction=task.language_instruction, control_hz=task.control_hz,
                  action_space=task.action_space, gripper_action_space=task.gripper_action_space,
                  edit_action_xyzg=task.edit_action_xyzg,
                  critic_camera_keys=list(task.critic_camera_keys) if hasattr(task, 'critic_camera_keys')
                      else ['base_0_rgb', 'left_wrist_0_rgb']),
        replan_steps=replan_steps, num_robot=num_robot)


def read_record(step):
    path = Path(step)/CONFIG_ITEM/CONFIG_FILE
    if not path.is_file():
        raise ValueError(f'Missing online config: {path}. Explicit legacy migration is required.')
    record = json.loads(path.read_text())
    if record.get('version') != 1 or record['config'].get('model_cls') != 'EXPOLearner':
        raise ValueError('Unsupported online checkpoint config')
    if record['factory_defaults'] != defaults_signature():
        raise ValueError('EXPO factory defaults changed; use matching source or migrate explicitly')
    sft_config.restore(record['sft'])
    return record


def restore_config(record, initial_sft_checkpoint=None):
    config = ConfigDict(unpack(record['config']))
    if initial_sft_checkpoint:
        config.initial_sft_checkpoint = str(initial_sft_checkpoint)
    _, recipe, norm_hash = load_sft(config)
    if recipe['resolved'] != record['sft']['resolved'] or norm_hash != record['norm_sha256']:
        raise ValueError('Initial SFT config/normalization differs from the online checkpoint')
    return config


def check_task(record, task, replan_steps):
    expected = record['task']
    for key, value in expected.items():
        actual = getattr(task, key, None)
        if key == 'critic_camera_keys':
            from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
            actual = list(getattr(task, key, CRITIC_CAMERA_KEYS))
        if actual != value:
            raise ValueError(f'Online checkpoint task mismatch: {key}')
    if replan_steps != record['replan_steps']:
        raise ValueError('replan_steps differs from checkpoint')


def validate_agent(record, agent):
    expected = getattr(agent.actor, 'checkpoint_record', None)
    if expected is None:
        raise ValueError('Construct the agent from checkpoint-owned model settings before restoring')
    # An asset path relocation is permitted; settings/content are checked independently.
    def comparable(r):
        r = json.loads(json.dumps(r))
        r['config'].pop('initial_sft_checkpoint', None)
        r.pop('policy_identity', None)
        r.pop('training', None)
        return r
    if comparable(record) != comparable(expected):
        raise ValueError('Online model configuration differs from saved checkpoint')


def configure_training(flags):
    """Resolve model flags before touching datasets, models, listeners or run outputs."""
    if flags.resume:
        root = Path(flags.output_dir)/flags.run_name/'checkpoints'
        steps = sorted((p for p in root.iterdir() if p.name.isdigit() and (p/'_CHECKPOINT_METADATA').exists()),
                       key=lambda p: int(p.name))
        if not steps:
            raise ValueError('No completed checkpoint to resume')
        if any(flags[name].present for name in flags if name == 'config' or name.startswith('config.')):
            raise ValueError('Resume reads saved model config; remove --config and its overrides')
        record = read_record(steps[-1])
        flags.config = restore_config(record, flags.initial_sft_checkpoint or None)
        for key in ('replan_steps', 'num_robot'):
            if key not in flags:
                if key == 'num_robot' and record[key] != 1:
                    raise ValueError('This trainer only supports one robot')
                continue
            if flags[key].present and getattr(flags, key) != record[key]:
                raise ValueError(f'Resume override differs from checkpoint: {key}')
            setattr(flags, key, record[key])
        for key, value in record.get('training', {}).items():
            if flags[key].present and getattr(flags, key) != value:
                raise ValueError(f'Resume learning setting differs: {key}')
            setattr(flags, key, value)
        check_task(record, flags.config_task, flags.replan_steps)
    else:
        if flags.initial_sft_checkpoint:
            flags.config.initial_sft_checkpoint = flags.initial_sft_checkpoint
    record = make_record(flags.config, flags.config_task, flags.replan_steps, getattr(flags, 'num_robot', 1))
    record['training'] = {key: getattr(flags, key) for key in
        ('batch_size', 'utd_ratio', 'offline_ratio', 'num_updates', 'update_type', 'step_interval', 'seed') if key in flags}
    return record
