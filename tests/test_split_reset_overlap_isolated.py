"""Stdlib-only control-flow tests; no application imports, sockets or hardware.

Extract the actual orchestration definitions and inject in-memory dependencies.
Run with /usr/bin/python3 -B tests/test_split_reset_overlap_isolated.py.
"""
import ast
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from copy import deepcopy
import json
import logging
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_definitions(path, names, namespace):
    nodes = []
    for node in ast.parse((ROOT / path).read_text()).body:
        if getattr(node, 'name', None) in names:
            # Function-local application imports must not initialize JAX/devices.
            node.body = [n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


class Array(list):
    def tolist(self):
        return list(self)


class SplitOverlapTests(unittest.TestCase):
    def test_deferred_ws_reset_reads_joints_but_not_camera(self):
        klass = next(n for n in ast.parse((ROOT/'client/envs/droid_env.py').read_text()).body
                     if getattr(n, 'name', None) == 'DroidEnv')
        klass.body = [n for n in klass.body if getattr(n, 'name', None) == 'reset']
        calls = []
        class RobotEnv:
            def reset(self, **kwargs):
                calls.append('motion')
        ns = dict(RobotEnv=RobotEnv, np=NS(asarray=Array), logging=logging, __name__=__name__)
        exec(compile(ast.Module(body=[klass], type_ignores=[]), '<reset>', 'exec'), ns)
        env = ns['DroidEnv']()
        env.reset_random = True
        env._before_reset = lambda: calls.append('before')
        env._robot = NS(get_joint_positions=lambda: calls.append('joints') or [0] * 7)
        env.prev_obs = {'robot_state': {'joint_positions': [0] * 7}}
        env.get_observation = lambda: calls.append('camera') or {'frame': 1}
        self.assertIsNone(env.reset(return_observation=False))
        self.assertEqual(calls, ['before', 'motion', 'joints'])
        self.assertEqual(env._raw_frame_buffer, [])
        self.assertEqual(env._steps_since_reset, 0)
        self.assertEqual(env.reset(), {'frame': 1})
        self.assertEqual(calls[-3:], ['before', 'motion', 'camera'])

    def test_collector_keeps_mirror_and_streaming_without_second_reset(self):
        physical, streamed, endings = [], [], []
        class Env:
            def __init__(self, index):
                self.index = index
            def reset(self):
                raise AssertionError('second reset')
            def start_episode(self):
                return {'initial': True}
            def step(self, action):
                physical.append((self.index, action))
                return action, 'policy'
            def get_observation(self):
                return {'initial': False}
            def get_info_for_step(self):
                return True, True, 1., 0.
            def close(self):
                pass
        ns = dict(deque=deque, Future=Future, ThreadPoolExecutor=ThreadPoolExecutor, deepcopy=deepcopy,
                  queue=queue, threading=threading, time=time, np=NS(asarray=lambda x: x),
                  canonical_observation=lambda obs, mirror: dict(obs, mirrored=mirror),
                  physical_action=lambda action, mirror: [-a for a in action] if mirror else action)
        load_definitions('expo_ft/utils/robot_round.py', ['collect_round'], ns)
        result = ns['collect_round']([Env(0), Env(1)], lambda obs: [[2.]], 1, 10000,
            mirror_robot=1, reset_done=True, check_session=lambda: None,
            on_transition=lambda *args: streamed.append(args), on_episode_end=lambda *args: endings.append(args))
        self.assertEqual(sorted(physical), [(0, [2.]), (1, [-2.])])
        self.assertEqual(len(streamed), 2)
        self.assertEqual(len(endings), 2)
        for index, (records, success) in enumerate(result):
            self.assertTrue(success)
            self.assertEqual(records[0]['actions'], [2.])
            self.assertEqual(records[0]['observations'], {'initial': True, 'mirrored': bool(index)})

    def run_pair(self, *, slow_reset=False, fail_reset=False, fail_update=False,
                 num_robot=2, start_step=0, max_steps=24, warmup=10,
                 checkpoint_buffer=False, fail_save=False, manual_save=False,
                 checkpoint_interval=2000):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        checkpoint_dir = Path(tmp.name)
        save_request = checkpoint_dir / 'save.request'
        messages = [{}, {}]
        lock = threading.Condition()
        events = []
        reset_started = [threading.Event() for _ in range(num_robot)]
        installed_new = threading.Event()
        update_entered = threading.Event()
        closed = threading.Event()
        envs = []
        errors = queue.Queue()
        self.test_thread = threading.current_thread()
        test = self
        contract = {'test': 'same identity'}
        saved = [0] * num_robot

        class Peer:
            def __init__(self, index):
                self.index = index
            def send(self, topic, key, value):
                with lock:
                    if topic != 'abort':
                        self.check_session()
                    events.append(('send', self.index, topic, key))
                    messages[1 - self.index][topic, key] = deepcopy(value)
                    lock.notify_all()
            def send_buffer(self, topic, key, snapshot):
                self.send(topic, key, {'version': snapshot.version})
            def receive(self, topic, key, timeout=None, *, check=None):
                deadline = time.monotonic() + 3
                with lock:
                    while (topic, key) not in messages[self.index]:
                        self.check_session()
                        if check:
                            check()
                        if time.monotonic() >= deadline:
                            raise TimeoutError((self.index, topic, key))
                        lock.wait(.01)
                    self.check_session()
                    if check:
                        check()
                    return messages[self.index][topic, key]
            def receive_buffer(self, topic, key, timeout=None, *, check=None):
                value = self.receive(topic, key, check=check)
                return nullcontext(NS(version=value['version'], timings={}))
            def check_session(self):
                if ('abort', 'test') in messages[self.index]:
                    raise RuntimeError('peer aborted')
            def release(self, topic, key):
                with lock:
                    if checkpoint_buffer and self.index == 0:
                        round_number = int(key.split('/')[1])
                        test.assertTrue(all(n > round_number for n in saved))
                    events.append(('release', self.index, topic, key))
                    messages[self.index].pop((topic, key), None)
            def flush(self):
                pass
            def wait_sent(self, *args):
                pass
            def close(self):
                pass

        class Env:
            def __init__(self, index):
                self.index, self.resets, self.frames = index, 0, 0
                self.finished = False
                envs.append(self)
            def reset_only(self):
                self.resets += 1
                self.finished = False
                events.append(('reset', self.index, self.resets))
                if self.resets > 1:
                    permission = ('send', 0, 'prepare_reset', 'test/' + str(self.resets - 1))
                    test.assertIn(permission, events)
                if self.resets == warmup + 2:
                    reset_started[self.index].set()
                    if fail_reset and self.index == 0:
                        raise RuntimeError('reset failed')
                    if fail_update:
                        if not closed.wait(2):
                            raise AssertionError('reset RPC not released on abort')
                    elif slow_reset:
                        if not installed_new.wait(2):
                            raise AssertionError('policy installation waited for reset')
                    elif not update_entered.wait(2):
                        raise AssertionError('reset did not overlap update')
                self.finished = True
                events.append(('reset_done', self.index, self.resets))
            def start_episode(self):
                test.assertTrue(all(e.finished for e in envs))
                self.frames += 1
                events.append(('observe', self.index, self.frames))
                return {'frame': self.frames}
            def close(self):
                closed.set()

        class Agent:
            rng = Array([1, 2])
            updates = 0
            version = start_step
            def replace(self, **kwargs):
                self.__dict__.update(kwargs)
                return self
            def update(self, *args):
                if self.updates == 0:
                    if manual_save:
                        save_request.touch()
                    update_entered.set()
                    if not fail_reset:
                        for event in reset_started:
                            if not event.wait(2):
                                raise AssertionError('learner updated before reset permission was consumed')
                    if fail_update:
                        raise RuntimeError('update failed')
                self.updates += 1
                events.append(('update', self.updates))
                return self, {'loss': 0.1}
            def sample_actions(self, observation):
                test.assertIsNot(threading.current_thread(), test.test_thread)
                events.append(('sample', observation['frame'], self.version))
                return [self.version], self, {}

        def collect(envs, sample, replan, hz, *, mirror_robot, on_transition, on_episode_end,
                    check_session, reset_done):
            test.assertTrue(reset_done)
            test.assertEqual(mirror_robot, 1 if num_robot == 2 else None)
            for env in envs:
                check_session()
                obs = env.start_episode()
                action = sample(obs)
                record = dict(observations=obs, actions=action, rewards=0., dones=True)
                on_transition(env.index, 0, record)
                on_episode_end(env.index, 1, False)

        def receive_round(channel, session, round_id, version, count):
            episodes = []
            for index in range(count):
                end = channel.receive('episode_end', '{}/{}/{}'.format(session, round_id, index))
                record = channel.receive('transition', '{}/{}/{}/0'.format(session, round_id, index))
                test.assertEqual(end['version'], version)
                test.assertEqual(record['version'], version)
                episodes.append(([record['transition']], end['success']))
            return episodes

        def import_policy(agent, snapshot, identity, version):
            test.assertEqual(snapshot.version, version)
            agent.version = version
            events.append(('install', version))
            if version != start_step:
                installed_new.set()
            return agent

        def fake_numpy_array(value, **kwargs):
            return Array(value) if isinstance(value, (list, tuple)) else value

        def save_batch(path, records, *, start_step):
            robot = int(path.name.split('-')[-1])
            if fail_save and robot == num_robot - 1:
                raise RuntimeError('save failed')
            test.assertTrue(all('is_success' in record for record in records))
            saved[robot] += 1
            events.append(('saved_batch', robot, start_step, len(records)))

        namespace = dict(
            __file__=str(ROOT/'expo_ft/distributed/runner.py'),
            Path=Path, json=json, logging=logging, time=time,
            ThreadPoolExecutor=ThreadPoolExecutor, FIRST_COMPLETED=FIRST_COMPLETED, wait=wait,
            np=NS(asarray=fake_numpy_array, uint32=int, isfinite=lambda x: NS(all=lambda: True)),
            jax=NS(random=NS(PRNGKey=lambda _: Array([1, 2])), device_put=lambda x, _: x,
                   device_get=lambda x: x, block_until_ready=lambda x: x,
                   tree=NS(leaves=lambda x: list(x.values()),
                           map=lambda fn, x: {k: fn(v) for k, v in x.items()})),
            LearnerGroup=lambda: NS(size=1, leader=True, call=lambda fn: fn(), barrier=lambda _: None),
            local_value=fake_numpy_array,
            replicate=lambda value, _: value,
            identity=lambda *a: contract, task_contract=lambda *a: None,
            key=lambda *parts: '/'.join(map(str, parts)),
            export_policy=lambda agent, contract, version: nullcontext(NS(version=version, size=0)),
            import_policy=import_policy, receive_round=receive_round, collect_round=collect,
            save_replay_buffer_batch=save_batch,
            atomic_json=lambda path, value: events.append(('checkpoint_ledger',)),
            wandb=NS(log=lambda *a, **kw: None), EnvClientWrapper=None,
        )
        load_definitions('expo_ft/utils/robot_round.py', ['updates_for_round'], namespace)
        load_definitions('expo_ft/utils/log_utils.py', ['log_round_interventions'], namespace)
        load_definitions('expo_ft/distributed/runner.py', ['_abort', 'run_learner', 'run_inference'], namespace)
        peers = [Peer(0), Peer(1)]
        namespace['_channel'] = lambda flags: peers[flags.role]
        flags = dict(seed=1, split_session='test', num_robot=num_robot, client_host='', client_port=8102,
                     output_dir='unused', run_name='test', resume=False, max_steps=max_steps,
                     batch_size=1, split_warmup_episodes=warmup, num_updates=3, step_interval=1, utd_ratio=20, replan_steps=8,
                     checkpoint_buffer=checkpoint_buffer, checkpoint_model=checkpoint_buffer, checkpoint_interval=checkpoint_interval,
                     config_task=NS(example_action=[], control_hz=10))
        learner_flags, inference_flags = NS(role=0, **flags), NS(role=1, **flags)
        learner_agent, inference_agent = Agent(), Agent()
        def env_factory(**kwargs):
            test.assertTrue(kwargs['env_creation_request']['async_video'])
            return Env(kwargs['port'] - 8102)
        def run(role):
            try:
                if role == 0:
                    namespace['run_learner'](learner_flags, learner_agent,
                        [NS(insert=lambda _: events.append(('insert',))) for _ in range(num_robot)],
                        NS(next_batch=lambda rng: ({}, None, rng)),
                        NS(wait_until_finished=lambda: events.append(('checkpoint_drained', save_request.is_file()))),
                        checkpoint_dir, lambda *args: events.append(('model_checkpoint', args[-1])),
                        start_step, False, None, 1 if num_robot == 2 else None)
                else:
                    namespace['run_inference'](inference_flags, inference_agent,
                        env_factory=env_factory)
            except BaseException as exc:
                errors.put(exc)
        threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(4)
        self.assertTrue(all(not t.is_alive() for t in threads), 'role did not exit')
        failures = list(errors.queue)
        if fail_save:
            self.assertTrue(any(str(e) == 'save failed' for e in failures), failures)
            self.assertFalse(any(e[0] in ('insert', 'update', 'model_checkpoint')
                                 or e[:2] == ('release', 0) for e in events))
            self.assertFalse(any(e[:3] == ('send', 0, 'prepare_reset') for e in events))
        elif fail_update or fail_reset:
            self.assertTrue(failures)
            self.assertTrue(any(str(e) == ('reset failed' if fail_reset else 'update failed') for e in failures), failures)
            self.assertTrue(all(env.frames == warmup + 1 for env in envs))
            self.assertTrue(closed.is_set())
        else:
            self.assertEqual(failures, [])
            if manual_save:
                self.assertFalse(save_request.exists())
            rounds = (max_steps - start_step) // num_robot
            self.assertTrue(all(env.resets == rounds and env.frames == rounds for env in envs))
            if rounds:
                self.assertEqual(learner_agent.updates, 6)
                samples = [e for e in events if e[0] == 'sample']
                self.assertTrue(all(e[2] == start_step for e in samples if e[1] <= warmup + 1))
                self.assertTrue(all(e[2] == (warmup + 1) * num_robot + start_step for e in samples if e[1] == warmup + 2))
            else:
                self.assertEqual(envs, [])
        return events

    def test_reset_overlaps_update_and_policy_wait(self):
        self.run_pair()

    def test_one_warmup_round_then_updates(self):
        self.run_pair(warmup=1, max_steps=6)

    def test_manual_checkpoint_after_updates_without_automatic_saving(self):
        events = self.run_pair(warmup=1, max_steps=6, manual_save=True)
        saved = events.index(('model_checkpoint', 4))
        drained = events.index(('checkpoint_drained', True))
        self.assertLess(events.index(('update', 3)), saved)
        self.assertLess(saved, drained)
        self.assertLess(drained, events.index(('update', 4)))
        self.assertEqual([e for e in events if e[0] == 'model_checkpoint'], [('model_checkpoint', 4)])

    def test_manual_and_interval_checkpoint_share_one_save(self):
        events = self.run_pair(warmup=1, max_steps=6, manual_save=True,
                               checkpoint_buffer=True, checkpoint_interval=4)
        self.assertEqual([e for e in events if e[0] == 'model_checkpoint'],
                         [('model_checkpoint', 4), ('model_checkpoint', 6)])

    def test_policy_ready_first_still_waits_for_both_resets(self):
        self.run_pair(slow_reset=True)

    def test_reset_failure_aborts_before_next_observation(self):
        self.run_pair(fail_reset=True)

    def test_learner_failure_closes_pending_reset(self):
        self.run_pair(fail_update=True)

    def test_single_robot_split_uses_same_handshake(self):
        self.run_pair(num_robot=1, max_steps=12)

    def test_stop_at_existing_step_does_not_create_robots(self):
        self.run_pair(start_step=24, max_steps=24)

    def test_both_batches_saved_before_release_reset_and_checkpoint(self):
        events = self.run_pair(checkpoint_buffer=True, warmup=1, max_steps=6)
        saved = [e for e in events if e[0] == 'saved_batch']
        self.assertEqual([(e[1], e[2], e[3]) for e in saved],
                         [(0, 1, 1), (1, 2, 1), (0, 3, 1), (1, 4, 1), (0, 5, 1), (1, 6, 1)])
        self.assertIn(('model_checkpoint', 6), events)
        for index, event in enumerate(events):
            if event[:3] == ('send', 0, 'prepare_reset'):
                completed = int(event[3].split('/')[1])
                self.assertEqual(sum(e[0] == 'insert' for e in events[:index]), completed * 2)

    def test_batch_save_failure_does_not_release_or_admit_round(self):
        self.run_pair(checkpoint_buffer=True, fail_save=True)

    def test_channel_wait_checks_background_failure(self):
        namespace = dict(time=time, message_id=lambda *a: 'test')
        # Extract only _wait; no Channel imports, sockets, or sidecar.
        klass = next(n for n in ast.parse((ROOT/'expo_ft/distributed/channel.py').read_text()).body
                     if getattr(n, 'name', None) == 'Channel')
        method = next(n for n in klass.body if getattr(n, 'name', None) == '_wait')
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<wait>', 'exec'), namespace)
        calls = []
        def check():
            calls.append('check')
            raise RuntimeError('reset failed')
        peer = NS(timeout=1, _rpc=lambda *a, **kw: calls.append('rpc'))
        with self.assertRaisesRegex(RuntimeError, 'reset failed'):
            namespace['_wait'](peer, 'receive', 'admit', 'test', None, check)
        self.assertEqual(calls, ['check'])


if __name__ == '__main__':
    unittest.main()
