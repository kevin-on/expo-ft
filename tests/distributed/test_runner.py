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
        for count in (1, 2):
            for saved_progress in (False, True):
                with self.subTest(num_robot=count, saved_progress=saved_progress):
                    self.check_resume(count, saved_progress)

    def check_resume(self, count, saved_progress):
        admissions = []
        releases = []
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
            def send(self, topic, key, value):
                if topic == 'admit':
                    admissions.append(value)
            def send_buffer(self, *args):
                pass
            def flush(self):
                pass
            def close(self):
                pass
            def release(self, topic, key):
                releases.append((topic, key))
            def receive(self, topic, key):
                if topic == 'installed':
                    return {'version': 44}
                if topic == 'episode_end':
                    return dict(version=44, length=3, success=True)
                if topic == 'transition':
                    index = int(key.rsplit('/', 1)[1])
                    row = record(index == 2)
                    if index == 1:
                        row.update(is_handoff=True, is_hil=False)
                    return dict(version=44, transition=row)
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
            buffers = [Buffer() for _ in range(count)]
            for step in range(1, 46):
                dest = path / f'robot-{((step-1)//2)%count}' / 'buffers' / f'{step:012d}.pkl'
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(pickle.dumps(record(step % 2 == 0, dummy=step == 1)))
            state = dict(identity={'test': True}, episode_count=22, pending_steps=0,
                         combine_rng=[1, 2], inference_rng=[3, 4], last_session='old', last_round=10)
            if saved_progress:
                state['rollout_progress'] = [dict(completed=22 // count, successes=7,
                    last='failure', transitions=44 // count) for _ in range(count)]
            (path / 'split-44.json').write_text(json.dumps(state))
            task = SimpleNamespace(control_hz=10, language_instruction='test', action_space='cartesian_velocity', gripper_action_space='velocity')
            flags = SimpleNamespace(split_session='new', config_task=task, num_robot=count, seed=42,
                replan_steps=8, max_steps=44 + 2 * count, num_updates=3, step_interval=1, batch_size=2, utd_ratio=1,
                split_warmup_episodes=10,
                checkpoint_buffer=True, checkpoint_model=True, checkpoint_interval=0)
            def batch(rng):
                self.assertEqual([len(b.rows) for b in buffers], [45] if count == 1 else [23, 24])
                return {}, None, rng
            manager = SimpleNamespace(wait_until_finished=lambda: None)
            saved = []
            patches.enter_context(patch.object(runner, '_channel', return_value=Peer()))
            patches.enter_context(patch.object(runner, 'identity', return_value={'test': True}))
            patches.enter_context(patch.object(runner, 'export_policy', return_value=__import__('contextlib').nullcontext(SimpleNamespace(size=100))))
            logs = patches.enter_context(patch('wandb.log'))
            with jax.default_device(jax.devices('cpu')[0]):
                agent = runner.run_learner(flags, Agent(jax.random.PRNGKey(1)), buffers,
                    SimpleNamespace(next_batch=batch), manager, path,
                    lambda m, a, s: saved.append((s, a.updates)), 44, True, None, 1 if count == 2 else None)
            self.assertEqual(agent.updates, 7)
            self.assertEqual(saved, [(44 + 2 * count, 7)])
            self.assertEqual(sum(topic == 'transition' for topic, _ in releases), 3 * count)
            self.assertTrue(all(not row.get('is_handoff') for buffer in buffers for row in buffer.rows))
            self.assertEqual(len(list((path / 'abandoned-replay/new').glob('robot-*/*.pkl'))), 1)
            ledger = json.loads((path / f'split-{44 + 2 * count}.json').read_text())
            self.assertEqual(ledger['episode_count'], 22 + count)
            self.assertEqual(ledger['inference_rng'], [9, 8])
            metrics = next(call.args[0] for call in logs.call_args_list if 'episodes' in call.args[0])
            self.assertEqual(metrics['robot-0/episode_length'], 3)
            self.assertAlmostEqual(metrics['robot-0/intervention_rate'], 2 / 3)
            self.assertEqual(admissions[0]['total_transitions'], 44)
            self.assertEqual(admissions[0]['completed_rounds'], 22 // count)
            for robot in range(count):
                prior = admissions[0]['rollout_progress'][robot]
                self.assertEqual(prior['completed'], 22 // count)
                self.assertEqual(prior['successes'], 7 if saved_progress else 22 // count)
                current = ledger['rollout_progress'][robot]
                self.assertEqual(current['completed'], prior['completed'] + 1)
                self.assertEqual(current['successes'], prior['successes'] + 1)
                self.assertEqual(current['transitions'], prior['transitions'] + 2)
                self.assertEqual(current['last'], 'success')
