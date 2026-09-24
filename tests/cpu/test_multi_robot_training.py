from types import SimpleNamespace

import jax
import numpy as np
import pytest

from expo_ft.utils import multi_robot_training as training
from test_robot_round import FakeEnv


@pytest.mark.parametrize("num_updates,expected_updates", [(3, 6), (0, 2)])
def test_training_driver_barrier_warmup_and_new_policy(monkeypatch, tmp_path, make_buffer,
                                                      num_updates, expected_updates):
    envs = [FakeEnv(3, 0), FakeEnv(5, 1)]
    monkeypatch.setattr(training, "EnvClientWrapper", lambda **kw: envs[kw["port"] - 8102])
    logs, checkpoints, update_versions = [], [], []
    monkeypatch.setattr(training.wandb, "log", lambda metrics, step: logs.append((step, dict(metrics))))
    buffers = [make_buffer(1), make_buffer(2)]
    class Agent:
        rng = jax.random.PRNGKey(1)
        version = 0
        def sample_actions(self, observation):
            return np.full((4, 2), self.version, np.float32), self, {}
        def replace(self, **kwargs):
            self.__dict__.update(kwargs)
            return self
        def update(self, agent, batch, utd_ratio, actor_batch):
            assert agent is self
            assert all(env.index == env.length for env in envs)
            assert len(buffers[0]) == 3 * len(envs[0].episodes)
            assert len(buffers[1]) == 5 * len(envs[1].episodes)
            self.version += 1
            update_versions.append(self.version)
            return self, {"loss": 0.1}
    flags = SimpleNamespace(seed=1, max_steps=56, replan_steps=2, client_host="localhost", client_port=8102,
                            config_task=SimpleNamespace(example_action=np.zeros(2), control_hz=10000),
                            num_updates=num_updates, step_interval=8, batch_size=4, utd_ratio=1,
                            checkpoint_buffer=True, checkpoint_model=True, checkpoint_interval=0)
    manager = SimpleNamespace(wait_until_finished=lambda: None)
    processor = SimpleNamespace(next_batch=lambda rng: ({}, None, rng))
    agent = Agent()
    training.train_multi_robot(flags, agent, buffers, processor, manager, tmp_path, tmp_path / "video",
                               lambda manager, agent, step: checkpoints.append(step), 0, False, None)
    assert len(update_versions) == expected_updates
    assert [metrics["updates"] for _, metrics in logs[:5]] == [0] * 5
    assert all(np.all(action == 0) for env in envs for episode in env.episodes[:6] for action in episode)
    first_update_count = 3 if num_updates else 1
    assert all(np.all(action == first_update_count) for env in envs for action in env.episodes[6])
    assert checkpoints == [56]
    assert (tmp_path / "round-56.json").exists()
    assert all(env.closed for env in envs)

    # Restore uses the checkpoint's cutoff and keeps each robot's episode history.
    abandoned = tmp_path / "robot-0/buffers/000000000057.pkl"
    abandoned.write_bytes(b"an unfinished round after the last checkpoint")
    restored = [make_buffer(1), make_buffer(2)]
    training.train_multi_robot(flags, agent, restored, processor, manager, tmp_path, tmp_path / "video",
                               lambda *_: None, 56, True, None)
    assert [len(buffer) for buffer in restored] == [21, 35]
    assert all(buffer.count_episodes_chronological() == 7 for buffer in restored)
    assert not abandoned.exists()
    next((tmp_path / "robot-0/buffers").glob("*.pkl")).unlink()
    with pytest.raises(ValueError, match="incomplete robot replay"):
        training.train_multi_robot(flags, agent, [make_buffer(1), make_buffer(2)], processor, manager,
                                   tmp_path, tmp_path / "video", lambda *_: None, 56, True, None)


def test_mirror_contract_camera_request_and_resume(monkeypatch, tmp_path, make_buffer):
    """The driver persists the same convention that it passes to the collector."""
    import json
    from pathlib import Path
    from conftest import transition

    requests, saved, collected = [], [], []
    def create_env(**kwargs):
        requests.append(kwargs['env_creation_request'])
        return FakeEnv(1)
    monkeypatch.setattr(training, 'EnvClientWrapper', create_env)
    monkeypatch.setattr(training.wandb, 'log', lambda *args, **kwargs: None)
    def collect(envs, sample, replan_steps, control_hz, mirror_robot=None):
        collected.append(mirror_robot)
        return [([transition(i, 0, done=True, success=True)], True) for i in range(2)]
    monkeypatch.setattr(training, 'collect_round', collect)
    flags = SimpleNamespace(seed=1, max_steps=2, replan_steps=2, client_host='localhost', client_port=8102,
                            config_task=SimpleNamespace(example_action=np.zeros(2), control_hz=10),
                            num_updates=3, step_interval=8, batch_size=4, utd_ratio=1,
                            checkpoint_buffer=True, checkpoint_model=True, checkpoint_interval=0)
    manager = SimpleNamespace(wait_until_finished=lambda: None)
    def run(mirror_robot, resuming):
        buffers = [make_buffer(1), make_buffer(2)]
        training.train_multi_robot(flags, object(), buffers, None, manager, tmp_path, tmp_path / 'video',
                                   lambda *args: saved.append(args[-1]), 2 if resuming else 0,
                                   resuming, None, mirror_robot=mirror_robot)
        return buffers
    run(1, False)
    assert collected == [1] and saved == [2]
    for i, request in enumerate(requests):
        cfg = json.loads((Path(__file__).resolve().parents[2] / f'configs/robots/robot-{i}-sft-eval.json').read_text())
        assert request['expected_camera_views'] == {key: cfg[key] for key in ('side_camera_id', 'wrist_camera_id')}
    ledger = json.loads((tmp_path/'round-2.json').read_text())
    assert ledger['mirror_robot'] == 1
    restored = run(1, True)
    assert [len(buffer) for buffer in restored] == [1, 1]
    requests.clear()
    for incompatible in (None, 0):
        with pytest.raises(ValueError, match='same live robot mirror convention'):
            run(incompatible, True)
    assert not requests  # reject before creating any live environment
    ledger.pop('mirror_robot')
    (tmp_path/'round-2.json').write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match='same live robot mirror convention'):
        run(1, True)
