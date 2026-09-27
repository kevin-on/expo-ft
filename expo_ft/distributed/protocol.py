"""Learning-level message identities and episode validation, without networking."""
import json
import os
from pathlib import Path
import struct

import numpy as np

from .channel import encode


def key(session, round_id, *parts):
    return '/'.join(map(str, (session, round_id) + parts))


def task_contract(flags, mirror_robot):
    views = []
    if mirror_robot is not None:
        for robot in range(flags.num_robot):
            path = Path(__file__).resolve().parents[2] / f'configs/robots/robot-{robot}.json'
            config = json.loads(path.read_text())
            views.append({k: config[k] for k in ('side_camera_id', 'wrist_camera_id')})
    return {'num_robot': flags.num_robot, 'mirror_robot': mirror_robot,
            'replan_steps': flags.replan_steps, 'control_hz': flags.config_task.control_hz,
            'language_instruction': flags.config_task.language_instruction,
            'action_space': flags.config_task.action_space,
            'gripper_action_space': flags.config_task.gripper_action_space,
            'camera_views': views,
            'camera_convention': 'robot0-side-right-wrist-left_robot1-side-left-wrist-right' if mirror_robot == 1 else 'task-default'}


def receive_round(channel, session, round_id, version, num_robot):
    result = []
    for robot in range(num_robot):
        end = channel.receive('episode_end', key(session, round_id, robot))
        if end['version'] != version or not isinstance(end['length'], int) or not 0 < end['length'] <= 100000:
            raise ValueError('invalid episode boundary/version')
        records = []
        for step in range(end['length']):
            message = channel.receive('transition', key(session, round_id, robot, step))
            if message['version'] != version:
                raise ValueError('mixed policy versions within a round')
            record = message['transition']
            if bool(record['dones']) != (step == end['length'] - 1):
                raise ValueError('missing or early terminal transition')
            if not np.isfinite(np.asarray(record['actions'])).all():
                raise ValueError('nonfinite executed action')
            records.append(record)
        result.append((records, bool(end['success'])))
    return result


def save_round(path, episodes, metadata):
    """One durable file per round, distinct from a model/replay checkpoint cursor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        for value in [metadata] + [{'robot': robot, 'success': success, 'transition': record}
                                 for robot, (records, success) in enumerate(episodes) for record in records]:
            payload = encode(value)
            f.write(struct.pack('!Q', len(payload)))
            f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
