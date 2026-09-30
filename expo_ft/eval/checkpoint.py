"""Eval-only weight envelope. Arrays and their metadata share one sealed RAM file."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import struct
import uuid

import numpy as np
import ml_dtypes  # Registers bfloat16 with NumPy.
import xxhash

from expo_ft.distributed.buffer import Buffer

HEADER = struct.Struct('<8sQQ')
MAGIC = b'EXPOEV02'
LIMIT = 16 * 1024**2


def leaves(tree, path=()):
    for name, value in sorted(tree.items()):
        if isinstance(value, dict):
            yield from leaves(value, path+(name,))
        else:
            yield path+(name,), value


def put(tree, path, value):
    for key in path[:-1]:
        tree = tree.setdefault(key, {})
    tree[path[-1]] = value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}', value):
        raise ValueError('Invalid run name or step')
    return value


def location(root, metadata):
    return within(root, metadata['checkpoint_path'])


def relative_path(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Expected root-relative path')
    path = Path(value)
    if path.is_absolute() or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('Expected root-relative path without traversal')
    return path


def within(root, value):
    root = Path(root).resolve()
    path = root / relative_path(value)
    if not path.resolve().is_relative_to(root):
        raise ValueError('Path escapes experiments root')
    return path


def common_path(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or path.parts[0] not in ('assets', 'model_config'):
        raise ValueError('Invalid common config/normalization path')
    return path


def pack(trees, metadata):
    buffer = Buffer.create()
    arrays = []
    try:
        with buffer.open() as f:
            f.write(bytes(64))
            for group, tree in sorted(trees.items()):
                for path, value in leaves(tree):
                    a = np.asarray(value)
                    if not a.flags.c_contiguous: a = np.ascontiguousarray(a)
                    if a.dtype.hasobject or a.dtype.fields or not a.dtype.isnative:
                        raise ValueError('Unsupported weight dtype')
                    offset = (f.tell()+63)//64*64
                    f.write(bytes(offset-f.tell()))
                    arrays.append(dict(group=group, path=path, shape=a.shape, dtype=a.dtype.name,
                                       offset=offset, nbytes=a.nbytes))
                    f.write(memoryview(a.reshape(-1).view(np.uint8)))
            offset = f.tell()
            tail = json.dumps(dict(metadata=metadata, arrays=arrays), sort_keys=True).encode()
            if len(tail) > LIMIT:
                raise ValueError('Oversized metadata')
            f.write(tail); f.seek(0); f.write(HEADER.pack(MAGIC, offset, len(tail)))
        buffer.seal()
        return buffer
    except BaseException:
        buffer.close()
        raise


def manifest(buffer):
    with buffer.view() as data:
        if len(data) < 64:
            raise ValueError('Truncated weights')
        magic, offset, length = HEADER.unpack_from(data)
        if magic != MAGIC or not 64 <= offset <= len(data) or length > LIMIT or offset+length != len(data):
            raise ValueError('Invalid weights header')
        result = json.loads(bytes(data[offset:]))
        end, seen = 64, set()
        for item in result['arrays']:
            key = (item['group'], tuple(item['path']))
            dtype = np.dtype(item['dtype'])
            shape = item['shape']
            if (not key[1] or any(not isinstance(k, str) or not k for k in key[1]) or key in seen
                    or dtype.hasobject or dtype.fields or not dtype.isnative
                    or not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                    or item['offset'] != (end+63)//64*64
                    or item['nbytes'] != math.prod(shape)*dtype.itemsize):
                raise ValueError('Invalid weight array metadata')
            end = item['offset']+item['nbytes']; seen.add(key)
            if end > offset:
                raise ValueError('Array outside payload')
        if end != offset:
            raise ValueError('Unexpected weight bytes')
        meta = result['metadata']
        if meta.get('kind') != 'base':
            identifier(meta['training_run_id']); relative_path(meta['checkpoint_path'])
            if meta.get('kind') not in ('sft', 'online') or meta.get('hash_algorithm') != 'xxh3_128':
                raise ValueError('Unknown evaluation format')
            for name, text in meta['files'].items():
                common_path(name)
                if not isinstance(text, str): raise ValueError('Config must be text')
        return result


@contextmanager
def arrays(buffer):
    info = manifest(buffer)
    trees = {}
    with buffer.view() as data:
        try:
            for item in info['arrays']:
                a = np.frombuffer(data, dtype=np.dtype(item['dtype']), count=math.prod(item['shape']),
                                  offset=item['offset']).reshape(item['shape'])
                put(trees.setdefault(item['group'], {}), item['path'], a)
            a = None
            yield trees, info['metadata']
        finally:
            trees.clear()


def fingerprint(tree):
    digest = xxhash.xxh3_128()
    for path, value in leaves(tree):
        a = np.asarray(value)
        if not a.flags.c_contiguous: a = np.ascontiguousarray(a)
        description = json.dumps([path, a.shape, a.dtype.name], separators=(',', ':')).encode()
        digest.update(struct.pack('<Q', len(description))); digest.update(description)
        digest.update(memoryview(a.reshape(-1).view(np.uint8)))
    return digest.hexdigest()


def read_file(path):
    b = Buffer.create(Path(path).stat().st_size)
    try:
        with Path(path).open('rb') as source, b.open() as target:
            while chunk := source.read(8*1024**2): target.write(chunk)
        b.seal(); manifest(b)
        return b
    except BaseException:
        b.close(); raise


def save(buffer, root):
    """Publish weights last; an existing checkpoint is never overwritten."""
    info = manifest(buffer)['metadata']
    step = location(root, info)
    folder = within(root, info['checkpoint_path']+'/eval'); folder.mkdir(parents=True, exist_ok=True)
    target = within(root, info['checkpoint_path']+'/eval/weights.bin')
    if target.exists():
        digest = xxhash.xxh3_128()
        with target.open('rb') as f:
            while chunk := f.read(8*1024**2): digest.update(chunk)
        if digest.hexdigest() != buffer.digest():
            raise ValueError('Different weights already saved at this run/step')
    for name, text in info['files'].items():
        dest = within(root, str(relative_path(info['checkpoint_path'])/common_path(name)))
        if dest.exists() and dest.read_text() != text:
            raise ValueError(f'Different common file already exists: {name}')
    for name, text in info['files'].items():
        dest = within(root, str(relative_path(info['checkpoint_path'])/common_path(name)))
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            _publish(dest, lambda f, text=text: f.write(text.encode()))
    if not target.exists():
        def write(f):
            with buffer.open() as source:
                while chunk := source.read(8*1024**2): f.write(chunk)
        _publish(target, write)
    return target


def _publish(path, write):
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('xb') as f:
            write(f); f.flush(); os.fsync(f.fileno())
        os.link(temp, path)  # No silent overwrite on concurrent save.
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        temp.unlink(missing_ok=True)


def restore_tree(path):
    """Read Orbax arrays onto host RAM, regardless of checkpoint device topology."""
    import jax
    import orbax.checkpoint as ocp
    with ocp.PyTreeCheckpointer() as checkpointer:
        meta = checkpointer.metadata(Path(path).resolve())
        return checkpointer.restore(Path(path).resolve(), args=ocp.args.PyTreeRestore(
            item=meta, restore_args=jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), meta)))


def pure(tree):
    from flax.traverse_util import flatten_dict, unflatten_dict
    flat = flatten_dict(tree)
    if flat and all(p[-1] == 'value' for p in flat):
        flat = {p[:-1]: v for p,v in flat.items()}
    return unflatten_dict(flat)


def split_actor(tree, config):
    from flax import nnx
    frozen = nnx.filterlib.to_predicate(config.freeze_filter)
    base, trained = {}, {}
    for path, value in leaves(tree):
        is_frozen = frozen(path, nnx.VariableState(nnx.Param, value))
        put(base if is_frozen else trained, path,
            np.asarray(value, dtype=ml_dtypes.bfloat16) if is_frozen else value)
    return base, trained


def export(checkpoint, kind, *, checkpoint_path, training_run_id, initial_sft=None, replan_steps=8):
    from openpi.training import checkpoint_config
    root = Path(checkpoint).resolve()
    relative_path(checkpoint_path)
    identifier(training_run_id)
    files = {}
    if kind == 'sft':
        recipe = checkpoint_config.read_record(root)
        cfg = checkpoint_config.restore(recipe)
        actor = pure(restore_tree(root/'params')['params'])
        files['assets/config.json'] = (root/'assets/config.json').read_text()
        norm_root = root
        extra = {}
        record = None
    else:
        from expo_ft.utils.model_config import read_record, restore_config
        record = read_record(root)
        config = restore_config(record, initial_sft)
        norm_root = Path(config.initial_sft_checkpoint)
        recipe = record['sft']; cfg = checkpoint_config.restore(recipe)
        files['model_config/config.json'] = (root/'model_config/config.json').read_text()
        # Initial SFT assets are retained at their normal relative paths.
        files['assets/config.json'] = (norm_root/'assets/config.json').read_text()
        params = restore_tree(root/'params')
        state = restore_tree(root/'agent')
        # When checkpoint params contain EMA, actual rollout parameters remain
        # in actor_train_state.params. Match online sampling, not EMA evaluation.
        rollout = state['actor_train_state']['params']
        actor = pure(rollout if rollout else params['actor_params'])
        extra = dict(encoder=params['batch_encoder_params'], edit=params['edit_actor_params'],
                     target_critic=state['target_critic']['params'])
        replan_steps = record['replan_steps']
    asset = cfg.data.assets.asset_id or cfg.data.repo_id
    name = 'assets/'+asset+'/norm_stats.json'; common_path(name)
    files[name] = (norm_root/name).read_text()
    if not 0 < replan_steps <= cfg.model.action_horizon: raise ValueError('Invalid replan steps')
    base, trained = split_actor(actor, cfg)
    meta = dict(kind=kind, training_run_id=training_run_id, checkpoint_path=checkpoint_path,
                hash_algorithm='xxh3_128', base_hash=fingerprint(base), files=files, replan_steps=replan_steps)
    return pack(dict(actor=trained, **extra), meta)


def load_base(params_path):
    """Load local base once on CPU. No model creation or GPU context."""
    from openpi.models.model import restore_params
    tree = restore_params(params_path, restore_type=np.ndarray, dtype=ml_dtypes.bfloat16)
    return pack({'base': tree}, {'kind': 'base'})


def config_and_norm(metadata):
    from openpi.training import checkpoint_config
    from openpi.shared.normalize import deserialize_json
    files = metadata['files']
    recipe = json.loads(files['assets/config.json'])
    cfg = checkpoint_config.restore(recipe)
    asset = cfg.data.assets.asset_id or cfg.data.repo_id
    stats = deserialize_json(files['assets/'+asset+'/norm_stats.json'])
    return cfg, stats


class BaseIdentity:
    """Cache the frozen-subset hash once per configured freeze filter."""
    def __init__(self):
        self.hashes = {}

    def __call__(self, base, payload):
        from openpi.training.checkpoint_config import describe
        meta = manifest(payload)['metadata']
        cfg, _ = config_and_norm(meta)
        key = json.dumps(describe(cfg.freeze_filter),sort_keys=True)
        if key not in self.hashes:
            with arrays(base) as (trees, _):
                frozen, _ = split_actor(trees['base'], cfg)
                self.hashes[key] = fingerprint(frozen)
                del frozen
        if self.hashes[key] != meta['base_hash']:
            raise ValueError('Local frozen base differs from checkpoint')


def validate_base(base, payload):
    BaseIdentity()(base, payload)
