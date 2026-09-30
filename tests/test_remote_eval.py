"""Device-free protocol/persistence tests; no environment or model constructors."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

from expo_ft.eval.checkpoint import pack, arrays, save, read_file, manifest, HEADER
from expo_ft.eval.server import Session, receiver, EvalProcess, prepare_output, lines


def payload(step='10', value=1):
    return pack({'actor':{'x':np.full((2,3),value,np.float32)}},dict(kind='sft',training_run_id='run-a',checkpoint_path='sft/run-a/checkpoints/'+step,
        hash_algorithm='xxh3_128',base_hash='base',files={'assets/config.json':'{}'},replan_steps=1))


class EnvelopeTest(unittest.TestCase):
    def test_canonical_local_weights_enable_eval_without_save_or_comparison(self):
        with tempfile.TemporaryDirectory() as folder, payload() as b:
            path=save(b,folder)
            s=Session(None,folder,validator=lambda *_:None)
            try:
                with mock.patch('expo_ft.eval.server.save',side_effect=AssertionError('Unexpected disk comparison')) as write:
                    s.load_saved(path)
                    self.assertEqual(s.saved,'Saved')
                    self.assertFalse(s.snapshot()['receive_armed'])
                    self.assertTrue(s.begin_eval());s.end_eval()
                    self.assertTrue(s.persist());write.assert_not_called()
                self.assertEqual(manifest(s.payload)['metadata']['checkpoint_path'],'sft/run-a/checkpoints/10')
            finally:s.close()

    def test_external_local_weights_need_copy_to_recorded_destination(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as destination, payload() as b:
            path=save(b,source)
            s=Session(None,destination,validator=lambda *_:None)
            try:
                s.load_saved(path)
                self.assertEqual(s.saved,'Loaded from disk');self.assertFalse(s.begin_eval())
                self.assertTrue(s.persist());self.assertTrue(s.begin_eval())
                self.assertTrue((Path(destination)/'sft/run-a/checkpoints/10/eval/weights.bin').is_file())
                s.end_eval()
            finally:s.close()

    def test_scalar_and_bfloat16_shapes(self):
        import ml_dtypes
        with pack({'base': {'scalar': np.asarray(3, np.float32),
                            'matrix': np.ones((2, 3), ml_dtypes.bfloat16)}},
                  {'kind': 'base'}) as b:
            with arrays(b) as (trees, _):
                self.assertEqual(trees['base']['scalar'].shape, ())
                self.assertEqual(trees['base']['matrix'].dtype, np.dtype(ml_dtypes.bfloat16))
                self.assertEqual(trees['base']['matrix'].shape, (2, 3))

    def test_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder, payload() as b:
            target=save(b,folder)
            self.assertEqual(target,Path(folder)/'sft/run-a/checkpoints/10/eval/weights.bin')
            self.assertEqual(save(b,folder),target)
            with read_file(target) as restored:
                self.assertEqual(restored.digest(),b.digest())
                with arrays(restored) as (trees,meta):
                    np.testing.assert_array_equal(trees['actor']['x'],np.ones((2,3)))
            with payload(value=2) as other:
                with self.assertRaisesRegex(ValueError,'Different weights'): save(other,folder)
            self.assertFalse(list(Path(folder).rglob('*.tmp')))

    def test_paths_and_truncated_data(self):
        for name in ('../bad','/tmp/bad','sft/../bad','sft//bad','sft/./bad'):
            with pack({},dict(kind='sft',training_run_id='run-a',checkpoint_path=name,hash_algorithm='xxh3_128',files={})) as b:
                with self.assertRaises(ValueError):manifest(b)
        from expo_ft.distributed.buffer import Buffer
        with Buffer.from_bytes(b'bad') as b:
            with self.assertRaises(ValueError):manifest(b)

    def test_failed_save_stays_ram_only(self):
        with tempfile.TemporaryDirectory() as folder:
            s=Session(None,folder,validator=lambda *_:None)
            s.toggle_receive();s.offer('a',10);s.accept('a',payload())
            with mock.patch('expo_ft.eval.server.save',side_effect=OSError('disk full')):s.persist()
            self.assertEqual(s.saved,'Save failed');self.assertEqual(s.mode,'receive')
            self.assertIsNotNone(s.payload);s.close()

    def test_modes_reject_receiving_during_eval_and_saving(self):
        with tempfile.TemporaryDirectory() as folder:
            s=Session(None,folder,validator=lambda *_:None)
            self.assertFalse(s.begin_eval())
            s.toggle_receive();self.assertTrue(s.offer('a',10));self.assertFalse(s.offer('b',10));self.assertFalse(s.begin_eval())
            s.accept('a',payload());self.assertTrue(s.persist());self.assertTrue(s.begin_eval())
            self.assertFalse(s.offer('b',10));self.assertFalse(s.persist())
            s.end_eval();self.assertEqual(s.saved,'Saved')
            s.persist();self.assertEqual(s.saved,'Saved')
            s.toggle_receive();s.offer('b',10);s.accept('b',payload('20'));self.assertEqual(s.saved,'RAM only');s.close()

    def test_receive_permission_is_single_use(self):
        with tempfile.TemporaryDirectory() as folder:
            s=Session(None,folder,validator=lambda *_:None)
            try:
                self.assertFalse(s.offer('locked',10))
                self.assertIn('Reception Locked', '\n'.join(lines(s,None,20)))
                self.assertTrue(s.toggle_receive())
                self.assertIn('Reception Ready for ONE transfer', '\n'.join(lines(s,None,20)))
                self.assertTrue(s.toggle_receive())  # withdraw unused permission
                self.assertFalse(s.offer('locked-again',10))
                s.toggle_receive();self.assertTrue(s.offer('first',10))
                self.assertFalse(s.toggle_receive());self.assertFalse(s.offer('second',10))
                s.accept('first',payload());first=s.payload
                self.assertFalse(s.offer('unexpected',10));self.assertIs(s.payload,first)
                # Failed/cancelled transfers require a fresh explicit permission too.
                s.toggle_receive();self.assertTrue(s.offer('cancelled',10))
                s.cancel('cancelled','Sender cancelled')
                self.assertFalse(s.offer('retry',10));self.assertIs(s.payload,first)
                s.toggle_receive();self.assertTrue(s.offer('expired',10))
                s.cancel('expired','Transfer timed out')
                self.assertFalse(s.offer('late',10))
                s.toggle_receive();self.assertTrue(s.persist())
                self.assertFalse(s.snapshot()['receive_armed'])
                s.toggle_receive();self.assertTrue(s.begin_eval())
                self.assertFalse(s.toggle_receive());s.end_eval()
                self.assertFalse(s.offer('after-eval',10))
            finally:s.close()

    def test_invalid_replacement_retains_previous(self):
        with tempfile.TemporaryDirectory() as folder:
            s=Session(None,folder,validator=lambda *_:None)
            s.toggle_receive();s.offer('a',10);s.accept('a',payload())
            first=s.payload
            s.validator=mock.Mock(side_effect=ValueError('wrong base'))
            s.toggle_receive();s.offer('b',10)
            with self.assertRaisesRegex(ValueError,'wrong base'):s.accept('b',payload('20'))
            s.cancel('b','wrong base');self.assertIs(s.payload,first)
            self.assertFalse(s.offer('retry',10));s.close()

    def test_saved_checkpoint_skips_io_until_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            s=Session(None,folder,validator=lambda *_:None)
            try:
                s.toggle_receive();s.offer('a',10);s.accept('a',payload());self.assertTrue(s.persist())
                self.assertIn('[E] Eval  [S]', '\n'.join(lines(s,None,20)))
                with mock.patch('expo_ft.eval.server.save',side_effect=AssertionError('Unexpected disk read')) as write:
                    self.assertTrue(s.persist())
                    for _ in range(2):
                        self.assertTrue(s.begin_eval());self.assertEqual(s.mode,'eval')
                        self.assertEqual(s.saved,'Saved');s.end_eval()
                    write.assert_not_called()
                # Same destination, different RAM payload must not inherit Saved.
                s.toggle_receive();s.offer('b',10);s.accept('b',payload(value=2))
                self.assertEqual(s.saved,'RAM only')
                self.assertIn('[E] Eval (disabled)', '\n'.join(lines(s,None,20)))
                with mock.patch('expo_ft.eval.server.save',wraps=save) as write:
                    self.assertFalse(s.begin_eval());write.assert_not_called()
                    self.assertFalse(s.persist());write.assert_called_once()
                self.assertEqual(s.saved,'Save failed');self.assertEqual(s.mode,'receive')
                self.assertIn('Different weights',s.notice)
            finally:s.close()

    def test_eval_save_failure_never_launches_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            s=Session(None,folder,validator=lambda *_:None)
            s.toggle_receive();s.offer('a',10);s.accept('a',payload())
            with mock.patch('expo_ft.eval.server.save',side_effect=OSError('disk full')) as write, \
                    mock.patch('expo_ft.eval.server.subprocess.Popen') as launch:
                with self.assertRaisesRegex(RuntimeError,'Save checkpoint'):EvalProcess(s,{})
                write.assert_not_called()
                self.assertFalse(s.persist())
                with self.assertRaisesRegex(RuntimeError,'Save checkpoint'):EvalProcess(s,{})
                write.assert_called_once()
            launch.assert_not_called()
            self.assertEqual(s.mode,'receive');self.assertEqual(s.saved,'Save failed')
            self.assertIsNotNone(s.payload);self.assertFalse((Path(folder)/'eval').exists());s.close()

    def test_path_preserved_and_separate_eval_record(self):
        with tempfile.TemporaryDirectory() as folder:
            s=Session(None,folder,validator=lambda *_:None)
            # Deliberately differs from the registered run ID and inferred layout.
            path='online/original-directory/checkpoints/special-step'
            meta=dict(kind='online',training_run_id='registered-id',checkpoint_path=path,
                hash_algorithm='xxh3_128',files={'assets/config.json':'original text'},replan_steps=8)
            s.toggle_receive();s.offer('a',10);s.accept('a',pack({},meta));self.assertTrue(s.persist());self.assertTrue(s.begin_eval())
            self.assertTrue((Path(folder)/path/'eval/weights.bin').is_file())
            out,opts=prepare_output(s,{'client_video_dir':'/unused/videos'})
            record=json.loads((out/'record.json').read_text())
            self.assertEqual(out.parent,Path(folder)/'eval')
            self.assertEqual(record,dict(run_id=out.name,training_run_id='registered-id',
                checkpoint_path=path,config_path=f'eval/{out.name}/config/eval_config.json'))
            self.assertEqual((Path(folder)/path/'assets/config.json').read_text(),'original text')
            self.assertEqual(opts['client_video_dir'],'/unused/videos/'+out.name)
            self.assertFalse(s.offer('b',10));s.end_eval();s.close()

    def test_save_and_eval_are_separate_exclusive_operations(self):
        with tempfile.TemporaryDirectory() as folder:
            s=Session(None,folder,validator=lambda *_:None)
            s.toggle_receive();s.offer('a',10);s.accept('a',payload())
            entered,release=threading.Event(),threading.Event()
            original=save
            def blocked(*args):
                entered.set();release.wait(5);return original(*args)
            with mock.patch('expo_ft.eval.server.save',side_effect=blocked):
                t=threading.Thread(target=s.persist);t.start()
                try:
                    self.assertTrue(entered.wait(5));self.assertEqual(s.saved,'Saving')
                    self.assertFalse(s.offer('b',10));self.assertFalse(s.begin_eval())
                finally:release.set();t.join()
            self.assertEqual(s.mode,'receive');self.assertEqual(s.saved,'Saved')
            self.assertTrue(s.begin_eval());self.assertFalse(s.offer('b',10))
            s.end_eval();s.close()

    def test_symlink_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside, payload() as b:
            (Path(folder)/'sft').symlink_to(outside,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'escapes'):save(b,folder)
            self.assertFalse(list(Path(outside).iterdir()))


# Reuse the established local RAM/TLS transport fixture; small payload only.
import importlib.util
_spec=importlib.util.spec_from_file_location("transport_fixture",Path(__file__).parent/"distributed/test_transport.py")
_fixture=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_fixture)
TransportTest=_fixture.TransportTest


class RemoteTransportTest(TransportTest):
    def test_offer_busy_replace_and_release(self):
        sender,recipient=self.channels
        s=Session(None,self.root/'runs',validator=lambda *_:None)
        stop=threading.Event();worker=threading.Thread(target=receiver,args=(recipient,s,stop));worker.start()
        try:
            sender.send('eval-offer','locked',dict(timeout=10));sender.flush()
            rejection=sender.receive('eval-admission','locked');sender.release('eval-admission','locked')
            self.assertFalse(rejection['accepted']);self.assertIn('press R',rejection['reason'])
            s.toggle_receive()
            sender.send('eval-offer','a',dict(timeout=10));sender.flush()
            self.assertTrue(sender.receive('eval-admission','a')['accepted']);sender.release('eval-admission','a')
            with payload() as b: sender.send_buffer('eval-weights','a',b)
            self.assertTrue(sender.receive('eval-result','a')['accepted']);sender.release('eval-result','a')
            sender.wait_sent('eval-weights','a')
            self.assertFalse(s.snapshot()['receive_armed'])
            sender.send('eval-offer','after-receive',dict(timeout=10));sender.flush()
            self.assertFalse(sender.receive('eval-admission','after-receive')['accepted'])
            sender.release('eval-admission','after-receive')
            self.assertTrue(s.persist());self.assertTrue(s.begin_eval())
            sender.send('eval-offer','b',dict(timeout=10));sender.flush()
            self.assertFalse(sender.receive('eval-admission','b')['accepted']);sender.release('eval-admission','b')
            s.end_eval()
        finally:stop.set();worker.join();s.close()
        self.assertIsNone(s.error)


if __name__=='__main__':unittest.main()
