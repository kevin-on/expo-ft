"""CPU/std-library-only orchestration tests; no JAX, sockets, devices or encoders.

Run: /usr/bin/python3 -B tests/test_local_reset_overlap_isolated.py
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import logging
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest

from test_split_reset_overlap_isolated import Array, load_definitions, ROOT


class LocalOverlapTests(unittest.TestCase):
    def run_local(self, *, slow_reset=False, fail_reset=None, fail_update=False,
                  fail_save=False, fail_close=False, rounds=12, start_step=0,
                  manual_save=False, checkpoint_interval=0):
        events, envs, requests, logs = [], [], [], []
        started = [threading.Event(), threading.Event()]
        update_started, updates_finished = threading.Event(), threading.Event()
        closed = threading.Event()
        caller = threading.get_ident()
        test = self
        saved, inserted = [0, 0], [0, 0]

        class Env:
            def __init__(self, index):
                self.index, self.resets, self.frames = index, 0, 0
                self.finished, self.closed = False, False
                envs.append(self)
            def reset_only(self):
                test.assertNotEqual(threading.get_ident(), caller)
                self.resets += 1
                self.finished = False
                events.append(('reset', self.index, self.resets))
                if self.resets > 1:
                    test.assertEqual(saved, [self.resets - 1] * 2)
                    test.assertEqual(inserted, saved)
                if self.resets == 12:
                    started[self.index].set()
                    if fail_reset is not None:
                        if self.index == fail_reset:
                            raise RuntimeError('reset failed')
                        test.assertTrue(closed.wait(2), 'close must release pending reset RPC')
                    elif fail_update:
                        test.assertTrue(closed.wait(2), 'update failure must close pending RPCs')
                    elif slow_reset and self.index == 1:
                        test.assertTrue(updates_finished.wait(2), 'reset must overlap all updates')
                    else:
                        test.assertTrue(update_started.wait(2), 'reset must overlap updates')
                self.finished = True
                events.append(('reset_done', self.index, self.resets))
            def start_episode(self):
                test.assertTrue(all(env.finished for env in envs))
                test.assertEqual(len({env.resets for env in envs}), 1)
                self.frames += 1
                test.assertEqual(self.frames, self.resets, 'no duplicate resets')
                if self.frames == 12:
                    test.assertTrue(updates_finished.is_set(), 'JAX work must finish before observation')
                events.append(('observe', self.index, self.frames))
                return {'frame': self.frames}
            def close(self):
                self.closed = True
                closed.set()
                if fail_close and self.index == 0:
                    raise RuntimeError('close failed')

        def env_factory(**kwargs):
            requests.append(kwargs['env_creation_request'])
            test.assertTrue(kwargs['lazy'])
            test.assertFalse(kwargs['recover'])
            return Env(kwargs['port'] - 8102)

        class Agent:
            rng = Array([1, 2])
            version = 0
            def replace(self, **kwargs):
                self.__dict__.update(kwargs)
                return self
            def sample_actions(self, obs):
                test.assertEqual(threading.get_ident(), caller)
                expected = 3 if obs['frame'] == 12 else 0
                test.assertEqual(self.version, expected)
                return [self.version], self, {}
            def update(self, *args):
                test.assertEqual(threading.get_ident(), caller)
                if self.version == 0:
                    if manual_save:
                        (Path(tmp)/'save.request').touch()
                    update_started.set()
                    for event in started:
                        test.assertTrue(event.wait(2), 'next reset must start during update')
                    if fail_update:
                        raise RuntimeError('update failed')
                self.version += 1
                return self, {'loss': .1}

        def block(agent):
            test.assertEqual(threading.get_ident(), caller)
            updates_finished.set()
            events.append(('updates_finished', agent.version))

        def collect(envs, sample, *args, mirror_robot=None, reset_done=False):
            test.assertTrue(reset_done)
            test.assertEqual(mirror_robot, 1)
            result = []
            for env in envs:
                obs = env.start_episode()
                action = sample(obs)
                result.append(([dict(observations=obs, actions=action, rewards=1., dones=True)], True))
            return result

        def save(path, records, *, start_step):
            if fail_save:
                raise RuntimeError('save failed')
            index = int(path.name[-1])
            test.assertTrue(all(record['is_success'] for record in records))
            saved[index] += len(records)

        def insert(index, record):
            test.assertTrue(record['is_success'])
            inserted[index] += 1

        ns = dict(__file__=str(ROOT/'expo_ft/utils/multi_robot_training.py'),
                  Path=Path, json=json, logging=logging,
                  ThreadPoolExecutor=ThreadPoolExecutor, FIRST_COMPLETED=FIRST_COMPLETED, wait=wait,
                  jax=NS(random=NS(PRNGKey=lambda seed: Array([seed, 0])), device_put=lambda x, _: x,
                         device_get=lambda x: x, block_until_ready=block),
                  np=NS(asarray=lambda x: x), EnvClientWrapper=env_factory, collect_round=collect,
                  save_replay_buffer_batch=save,
                  wandb=NS(log=lambda metrics, step: logs.append((step, dict(metrics)))))
        load_definitions('expo_ft/utils/robot_round.py', ['updates_for_round'], ns)
        load_definitions('expo_ft/utils/log_utils.py', ['log_round_interventions'], ns)
        load_definitions('expo_ft/utils/multi_robot_training.py', ['train_multi_robot'], ns)
        flags = NS(seed=1, max_steps=2 * rounds, replan_steps=8, client_host='', client_port=8102,
                   config_task=NS(example_action=[], control_hz=10), batch_size=1,
                   num_updates=3, step_interval=1, utd_ratio=20,
                   checkpoint_buffer=True, checkpoint_model=True, checkpoint_interval=checkpoint_interval)
        agent = Agent()
        with tempfile.TemporaryDirectory() as tmp:
            def run():
                ns['train_multi_robot'](flags, agent,
                    [NS(insert=lambda r, i=i: insert(i, r)) for i in range(2)],
                    NS(next_batch=lambda rng: ({}, None, rng)),
                    NS(wait_until_finished=lambda: events.append(('checkpoint_drained',))),
                    Path(tmp), Path(tmp)/'video', lambda *args: events.append(('checkpoint', args[-1])),
                    start_step, False, None, mirror_robot=1)
            if fail_reset is not None or fail_update or fail_save:
                message = 'reset failed' if fail_reset is not None else 'update failed' if fail_update else 'save failed'
                with self.assertRaisesRegex(RuntimeError, message):
                    run()
                self.assertTrue(all(env.frames == (1 if fail_save else 11) for env in envs))
            else:
                if fail_close:
                    with self.assertLogs(level='ERROR'):
                        run()
                else:
                    run()
                if manual_save:
                    self.assertFalse((Path(tmp)/'save.request').exists())
                    self.assertEqual([e for e in events if e[0] == 'checkpoint'],
                                     [('checkpoint', 22), ('checkpoint', 24)])
                    saved_event = events.index(('checkpoint', 22))
                    self.assertLess(events.index(('updates_finished', 3)), saved_event)
                    self.assertEqual(events[saved_event + 1], ('checkpoint_drained',))
                for env in envs:
                    self.assertEqual(env.resets, rounds - start_step // 2)
                    self.assertEqual(env.frames, env.resets)
                self.assertEqual(agent.version, 6 if rounds == 12 else 0)
                if logs:
                    self.assertEqual([m['updates'] for _, m in logs[:10]], [0] * min(10, rounds))
                    ledger = json.loads((Path(tmp)/f'round-{2 * rounds}.json').read_text())
                    self.assertEqual(ledger['mirror_robot'], 1)
                    self.assertEqual(ledger['episode_count'], 2 * rounds)
            self.assertTrue(all(env.closed for env in envs))
            self.assertEqual(events[-1], ('checkpoint_drained',))
            self.assertTrue(all(r['async_video'] for r in requests))
            for i, request in enumerate(requests):
                cfg = json.loads((ROOT/f'configs/robots/robot-{i}.json').read_text())
                self.assertEqual(request['expected_camera_views'],
                                 {k: cfg[k] for k in ('side_camera_id', 'wrist_camera_id')})

    def test_resets_overlap_update_and_observation_uses_new_policy(self):
        self.run_local()

    def test_manual_checkpoint_after_updates_and_training_continues(self):
        self.run_local(manual_save=True)

    def test_manual_checkpoint_coinciding_with_interval_is_not_duplicated(self):
        self.run_local(manual_save=True, checkpoint_interval=22)

    def test_slow_second_reset_blocks_both_observations(self):
        self.run_local(slow_reset=True)

    def test_second_reset_failure_interrupts_pending_first_reset(self):
        self.run_local(fail_reset=1)

    def test_first_reset_failure_interrupts_pending_second_reset(self):
        self.run_local(fail_reset=0)

    def test_update_failure_closes_pending_reset_rpcs(self):
        self.run_local(fail_update=True)

    def test_replay_save_failure_does_not_start_next_reset(self):
        self.run_local(fail_save=True, rounds=3)

    def test_no_reset_after_final_warmup_round(self):
        self.run_local(rounds=3)

    def test_no_motion_when_already_at_max_steps(self):
        self.run_local(rounds=0, start_step=0)

    def test_close_failure_still_closes_other_robot_and_drains_checkpoint(self):
        self.run_local(rounds=1, fail_close=True)


if __name__ == '__main__':
    unittest.main()
