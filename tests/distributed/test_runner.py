"""Barrier/cursor checks without a model, robot, camera, or network connection."""
from contextlib import ExitStack
from dataclasses import dataclass, replace
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import numpy as np

# Match train_pi_robo's import order; the existing eager agents/data packages
# otherwise form a cycle when the replay module is imported first.
import expo_ft.agents  # noqa: F401
from expo_ft.distributed import runner


class ResumeTest(unittest.TestCase):
    def test_resume_uses_checkpoint_cursor_and_default_dummy_filter(self):
        @dataclass
        class Agent:
            rng: object
            updates: int = 4
            replace = replace
            def update(self, learner, batch, utd, actor_batch):
                return replace(self, updates=self.updates + 1), {'loss': np.float32(1)}
        class Buffer:
            def __init__(self):
                self.rows = []
            def insert(self, record):
                self.rows.append(record)
            def restore_success_marks(self):
                pass
        class Peer:
            def send(self, *args):
                pass
            def send_buffer(self, *args):
                pass
            def flush(self):
                pass
            def close(self):
                pass
            def release(self, *args):
                pass
            def receive(self, topic, key):
                if topic == 'installed':
                    return {'version': 44}
                if topic == 'episode_end':
                    return dict(version=44, length=2, success=True)
                if topic == 'transition':
                    return dict(version=44, transition=record(int(key.rsplit('/', 1)[1]) == 1))
                if topic == 'round_finished':
                    return dict(version=44, inference_rng=[9, 8])
                if topic == 'stopped':
                    return {'stopped': True}
                raise AssertionError(topic)
        def record(done=False, dummy=False):
            return dict(actions=-np.ones(7) if dummy else np.zeros(7), observations={}, rewards=0., masks=float(not done),
                        dones=done, is_hil=True, is_success=True)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
            path = Path(directory)
            buffers = [Buffer(), Buffer()]
            for step in range(1, 46):
                dest = path / f'robot-{((step-1)//2)%2}' / 'buffers' / f'{step:012d}.pkl'
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(pickle.dumps(record(step % 2 == 0, dummy=step == 1)))
            state = dict(identity={'test': True}, episode_count=22, pending_steps=0,
                         combine_rng=[1, 2], inference_rng=[3, 4], last_session='old', last_round=10)
            (path / 'split-44.json').write_text(json.dumps(state))
            task = SimpleNamespace(control_hz=10, language_instruction='test', action_space='cartesian_velocity', gripper_action_space='velocity')
            flags = SimpleNamespace(split_session='new', config_task=task, num_robot=2, seed=42,
                replan_steps=8, max_steps=48, num_updates=3, step_interval=1, batch_size=2, utd_ratio=1,
                split_warmup_episodes=10,
                checkpoint_buffer=True, checkpoint_model=True, checkpoint_interval=0)
            def batch(rng):
                self.assertEqual([len(b.rows) for b in buffers], [23, 24])
                return {}, None, rng
            manager = SimpleNamespace(wait_until_finished=lambda: None)
            saved = []
            patches.enter_context(patch.object(runner, '_channel', return_value=Peer()))
            patches.enter_context(patch.object(runner, 'identity', return_value={'test': True}))
            patches.enter_context(patch.object(runner, 'export_policy', return_value=__import__('contextlib').nullcontext(SimpleNamespace(size=100))))
            patches.enter_context(patch('wandb.log'))
            with jax.default_device(jax.devices('cpu')[0]):
                agent = runner.run_learner(flags, Agent(jax.random.PRNGKey(1)), buffers,
                    SimpleNamespace(next_batch=batch), manager, path,
                    lambda m, a, s: saved.append((s, a.updates)), 44, True, None, 1)
            self.assertEqual(agent.updates, 7)
            self.assertEqual(saved, [(48, 7)])
            self.assertEqual(len(list((path / 'abandoned-replay/new').glob('robot-*/*.pkl'))), 1)
            ledger = json.loads((path / 'split-48.json').read_text())
            self.assertEqual(ledger['episode_count'], 24)
            self.assertEqual(ledger['inference_rng'], [9, 8])
