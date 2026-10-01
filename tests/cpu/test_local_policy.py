"""Tiny real JAX arrays: GPU-style replica selection without model/device access.

Run with JAX_PLATFORMS=cpu and two virtual CPU devices. No weights or robots.
"""
from dataclasses import dataclass, replace
import json
from types import SimpleNamespace as NS

from flax import nnx
import jax
import numpy as np
import pytest

from expo_ft.distributed.local import device_replica, inference_view, LocalSession, LocalPolicyExchange


@dataclass
class State:
    params: object
    step: object
    opt_state: object = 'optimizer'
    ema_params: object = None
    replace = replace


@dataclass
class Agent:
    actor: object
    rng: object
    actor_train_state: object
    batch_encoder: object
    edit_actor: object
    target_critic: object
    critic: object = 'learner critic'
    temp: object = 'temperature'
    target_actor_params: object = 'target actor'
    rollout_cache: bool = False
    _infer_cache: object = None
    replace = replace


def learner(devices):
    mesh = jax.sharding.Mesh(np.array(devices), ('batch',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    weight = jax.device_put(np.arange(4, dtype=np.float32), sharding)
    state = State({'weight': weight}, jax.device_put(np.int32(3), sharding))
    # Use the actual OpenPI parameter container type; selecting arrays must not
    # mutate the learner's nnx.State/VariableState wrappers.
    params = parameter_state(weight)
    actor_state = replace(state, params=params)
    actor = NS(infer_device=devices[0], infer_sharding=jax.sharding.SingleDeviceSharding(devices[0]))
    return Agent(actor, jax.random.PRNGKey(7), actor_state, state, state, state), weight


def parameter_state(weight):
    class Model(nnx.Module):
        def __init__(self):
            self.weight = nnx.Param(weight)
    return nnx.state(Model())


@pytest.mark.parametrize('device_count', [1, 2])
def test_inference_uses_existing_replica_and_no_optimizer(device_count):
    if len(jax.devices()) < device_count:
        pytest.skip('Expose two virtual CPU devices for the multi-device case')
    agent, weights = learner(jax.devices()[:device_count])
    rng = jax.random.PRNGKey(91)
    view = inference_view(agent, rng)
    original = next(s.data for s in weights.addressable_shards if s.device == agent.actor.infer_device)
    selected = view.actor_train_state.params['weight'].value
    assert selected.unsafe_buffer_pointer() == original.unsafe_buffer_pointer()
    assert view.batch_encoder.params['weight'].unsafe_buffer_pointer() == original.unsafe_buffer_pointer()
    assert view._infer_cache['actor_train_state'] is view.actor_train_state
    assert view.actor is agent.actor
    assert view.actor_train_state.opt_state == view.batch_encoder.opt_state == ()
    assert view.critic is view.temp is view.target_actor_params is None
    assert agent.actor_train_state.opt_state == 'optimizer'
    assert agent.actor_train_state.params['weight'].value is weights
    np.testing.assert_array_equal(view.rng, rng)
    np.testing.assert_array_equal(agent.rng, jax.random.PRNGKey(7))
    # An actual inference operation sees identical inputs/weights.
    infer = jax.jit(lambda p, x: p @ x)
    np.testing.assert_array_equal(infer(selected, selected), infer(original, original))


def test_partial_parameter_shards_are_rejected_without_gather():
    if len(jax.devices()) < 2:
        pytest.skip('Expose two virtual CPU devices')
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ('batch',))
    split = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('batch'))
    weights = jax.device_put(np.arange(4, dtype=np.float32), split)
    with pytest.raises(ValueError, match='replicated parameters'):
        device_replica(weights, jax.devices()[0])


def test_policy_refresh_after_donated_update_preserves_inference_rng():
    agent, weights = learner(jax.devices()[:1])
    session = LocalSession(1)
    left, right = session.endpoint(0), session.endpoint(1)
    policy = LocalPolicyExchange(left, right, 's', 42)
    contract = {'test': True}
    assert policy.publish(agent, contract, 0)['policy_bytes'] == 0
    view, _ = policy.install(None, contract, 0)
    right.release('policy', 's/0')
    np.testing.assert_array_equal(view.rng, jax.random.fold_in(jax.random.PRNGKey(42), 7351))
    _, advanced_rng = jax.random.split(view.rng)
    view = view.replace(rng=advanced_rng)
    update = jax.jit(lambda x: x + 1, donate_argnums=(0,))
    new_weights = update(weights)
    new_weights.block_until_ready()
    new_state = replace(agent.batch_encoder, params={'weight': new_weights})
    params = parameter_state(new_weights)
    updated = replace(agent, actor_train_state=replace(agent.actor_train_state, params=params),
                      batch_encoder=new_state, edit_actor=new_state, target_critic=new_state)
    policy.publish(updated, contract, 2)
    current, _ = policy.install(view, contract, 2)
    right.release('policy', 's/2')
    np.testing.assert_array_equal(current.actor_train_state.params['weight'].value, np.arange(4) + 1)
    np.testing.assert_array_equal(current.rng, advanced_rng)
    np.testing.assert_array_equal(updated.rng, jax.random.PRNGKey(7))
    assert session.messages == [{}, {}]


def test_policy_rejects_wrong_version_and_contract():
    agent, _ = learner(jax.devices()[:1])
    session = LocalSession(1)
    left, right = session.endpoint(0), session.endpoint(1)
    policy = LocalPolicyExchange(left, right, 's', 42)
    policy.publish(agent, {'a': 1}, 0)
    with pytest.raises(ValueError, match='identity/version'):
        policy.install(None, {'a': 2}, 0)


@pytest.mark.parametrize('num_robot', [1, 2])
@pytest.mark.parametrize('device_count', [1, 2])
def test_colocated_rounds_and_resume_with_actual_runners(tmp_path, monkeypatch, num_robot, device_count):
    from expo_ft.distributed import local, policy, runner
    import wandb

    # Only the model, hardware endpoint and W&B sink are fake. The supervisor,
    # mailbox, device view, collector, split protocol, replay files and ledger run.
    sampled_rngs, sampled_weights, logged, saved = [], [], [], []
    def sample(self, observation):
        sampled_rngs.append(np.asarray(self.rng).tolist())
        value = float(self.actor_train_state.params['weight'].value[0])
        sampled_weights.append(value)
        return np.full((1, 7), value), replace(self, rng=jax.random.split(self.rng)[0]), {}
    def update(self, agent, batch, utd, actor_batch):
        weights = self.actor_train_state.params['weight'].value + 1
        state = replace(self.batch_encoder, params={'weight': weights})
        actor = replace(self.actor_train_state, params=parameter_state(weights),
                        step=self.actor_train_state.step + 1)
        return replace(self, actor_train_state=actor, batch_encoder=state,
                       edit_actor=state, target_critic=state,
                       rng=jax.random.split(self.rng)[0]), {'loss': np.float32(1)}
    monkeypatch.setattr(Agent, 'sample_actions', sample, raising=False)
    monkeypatch.setattr(Agent, 'update', update, raising=False)
    monkeypatch.setattr(policy, 'identity', lambda *a: {'robots': num_robot})
    monkeypatch.setattr(wandb, 'log', lambda data, **kw: logged.append(dict(data)))
    monkeypatch.setattr(runner, 'build_inference', lambda *a: pytest.fail('second model build'))
    monkeypatch.setattr(runner, 'export_policy', lambda *a: pytest.fail('snapshot export'))
    monkeypatch.setattr(runner, 'import_policy', lambda *a: pytest.fail('snapshot import'))
    monkeypatch.setattr(runner, '_channel', lambda *a: pytest.fail('network transport'))
    class Buffer:
        def __init__(self):
            self.rows = []
        def insert(self, record):
            self.rows.append(record)
        def restore_success_marks(self):
            pass
    class Env:
        def __init__(self, **kwargs):
            self.robot = kwargs['port'] - 8102
            self.index = 0
            assert kwargs['env_creation_request']['async_video']
            assert set(kwargs['env_creation_request']['expected_camera_views']) == {'side_camera_id', 'wrist_camera_id'}
        def reset_only(self):
            self.index = 0
        def start_episode(self):
            return self.get_observation()
        def get_observation(self):
            result = {key: np.zeros((2, 2, 3), np.uint8) for key in (
                'exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left')}
            result.update(cartesian_position=np.zeros(6, np.float32), gripper_position=np.zeros(1))
            return result
        def step(self, action):
            self.index += 1
            return action, 'policy'
        def get_info_for_step(self):
            done = self.index == 2
            return done, done and self.robot == 0, float(done and self.robot == 0), float(not done)
        def close(self):
            pass
    if len(jax.devices()) < device_count:
        pytest.skip('Expose two virtual CPU devices')
    agent, _ = learner(jax.devices()[:device_count])
    task = NS(example_action=np.zeros((1, 7)), control_hz=10000,
              language_instruction='test', action_space='cartesian_velocity', gripper_action_space='velocity')
    flags = NS(fsdp_devices=1, split_timeout=3, split_session='first', seed=42,
               num_robot=num_robot, config_task=task, resume=False,
               client_host='unused', client_port=8102, output_dir=str(tmp_path), run_name='test',
               replan_steps=1, max_steps=6 * num_robot, split_warmup_episodes=1,
               batch_size=1, num_updates=1, step_interval=1, utd_ratio=1,
               checkpoint_model=True, checkpoint_buffer=True, checkpoint_interval=0)
    processor = NS(next_batch=lambda rng: ({}, None, rng))
    manager = NS(wait_until_finished=lambda: None)
    def run(agent, buffers, start=0, resuming=False):
        return local.run_colocated(flags, agent, buffers, processor, manager, tmp_path,
            lambda manager, agent, step: saved.append(step), start, resuming, None,
            1 if num_robot == 2 else None, env_factory=Env)
    buffers = [Buffer() for _ in range(num_robot)]
    result = run(agent, buffers)
    assert saved == [6 * num_robot]
    assert [len(b.rows) for b in buffers] == [6] * num_robot
    assert sampled_weights == [0.] * (4 * num_robot) + [1.] * (2 * num_robot)
    assert float(result.actor_train_state.params['weight'].value[0]) == 2
    assert all(data['robot-0/success_without_intervention'] == 1 for data in logged if 'episodes' in data)
    ledger = json.loads((tmp_path / f'split-{6 * num_robot}.json').read_text())
    assert ledger['episode_count'] == 3 * num_robot
    flags.split_session, flags.resume, flags.max_steps = 'resumed', True, 8 * num_robot
    restored = [Buffer() for _ in range(num_robot)]
    sample_count = len(sampled_rngs)
    resumed = run(result, restored, start=6 * num_robot, resuming=True)
    assert sampled_rngs[sample_count] == ledger['inference_rng']
    assert sampled_weights[sample_count:] == [2.] * (2 * num_robot)
    assert [len(b.rows) for b in restored] == [8] * num_robot
    assert saved == [6 * num_robot, 8 * num_robot]
    assert float(resumed.actor_train_state.params['weight'].value[0]) == 3
