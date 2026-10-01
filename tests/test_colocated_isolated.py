"""Real split runners + real in-process mailbox, fake models/environments only."""
import runpy
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace as NS
import unittest

import test_split_reset_overlap_isolated as overlap

ROOT = overlap.ROOT

LocalSession = runpy.run_path(str(ROOT / 'expo_ft/distributed/local.py'))['LocalSession']


class ColocatedRounds(overlap.SplitOverlapTests):
    # Run every split round regression through the colocated delivery too.
    def run_pair(self, **kwargs):
        return super().run_pair(local_delivery=True, **kwargs)


class LocalMailboxTests(unittest.TestCase):
    def test_records_are_owned_until_release(self):
        session = LocalSession(1)
        sender, receiver = session.endpoint(0), session.endpoint(1)
        record = {'actions': [1, 2], 'observations': {'frame': [3]}}
        sender.send('transition', 's/0/0/0', record)
        record['observations']['frame'][0] = 99
        self.assertEqual(receiver.receive('transition', 's/0/0/0')['observations']['frame'], [3])
        receiver.release('transition', 's/0/0/0')
        self.assertEqual(session.messages, [{}, {}])

    def test_policy_is_not_copied(self):
        session = LocalSession(1)
        policy = object()
        session.endpoint(0).send_reference('policy', 's/0', policy)
        self.assertIs(session.endpoint(1).receive('policy', 's/0'), policy)

    def test_abort_wakes_receiver_and_callbacks_run(self):
        session = LocalSession(5)
        checked = threading.Event()
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(session.endpoint(1).receive, 'admit', 's/0', check=checked.set)
            self.assertTrue(checked.wait(1))
            session.abort(ValueError('learner failed'))
            with self.assertRaisesRegex(RuntimeError, 'peer stopped'):
                pending.result(timeout=1)

    def test_peer_close_allows_queued_stop_ack_then_fails_missing_receive(self):
        session = LocalSession(1)
        learner, inference = session.endpoint(0), session.endpoint(1)
        inference.send('stopped', 's', {'stopped': True})
        inference.close()
        self.assertEqual(learner.receive('stopped', 's'), {'stopped': True})
        with self.assertRaisesRegex(RuntimeError, 'peer closed'):
            learner.receive('missing', 's')


class LocalSupervisorTests(unittest.TestCase):
    def run_roles(self, failure=None):
        namespace = runpy.run_path(str(ROOT / 'expo_ft/distributed/local.py'))
        events = []
        main_thread = threading.current_thread()
        def learner(*args, **kwargs):
            self.assertIsNot(threading.current_thread(), main_thread)
            channel = kwargs['channel']
            if failure == 'learner':
                raise ValueError('original learner error')
            channel.send('admit', 's/0', {})
            channel.receive('stopped', 's')
            events.append('learner_done')
            return 'latest_agent'
        def inference(flags, **kwargs):
            self.assertIs(threading.current_thread(), main_thread)
            self.assertNotIn('agent', kwargs)  # do not build another model
            channel = kwargs['channel']
            channel.receive('admit', 's/0')
            if failure == 'inference':
                raise ValueError('original inference error')
            channel.send('stopped', 's', {})
            events.append('inference_done')
        namespace.update(jax=NS(process_count=lambda: 1),
                         identity=lambda *a: {}, task_contract=lambda *a: {},
                         run_learner=learner, run_inference=inference)
        overlap.load_definitions('expo_ft/distributed/local.py', ['run_colocated'], namespace)
        flags = NS(fsdp_devices=1, split_timeout=1, split_session='s', seed=42)
        result = namespace['run_colocated'](flags, NS(rollout_cache=False), [], None, None,
            'unused', None, 0, False, None, None)
        self.assertEqual(result, 'latest_agent')
        self.assertEqual(set(events), {'learner_done', 'inference_done'})

    def test_roles_finish_and_main_thread_owns_inference(self):
        self.run_roles()

    def test_learner_error_is_propagated_and_peer_exits(self):
        with self.assertRaisesRegex(ValueError, 'original learner error'):
            self.run_roles('learner')

    def test_inference_error_is_propagated_and_peer_exits(self):
        with self.assertRaisesRegex(ValueError, 'original inference error'):
            self.run_roles('inference')


if __name__ == '__main__':
    unittest.main()
