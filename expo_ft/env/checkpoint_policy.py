"""Model-only checkpoint loaders shared by evaluation tools (no hardware imports)."""
from pathlib import Path
import numpy as np


class SFTPolicy:
    def __init__(self, checkpoint, *, seed=42, prompt=None):
        from openpi.training import checkpoint_config
        from openpi.policies.policy_config import create_trained_policy
        self.config = checkpoint_config.load(checkpoint)
        if not getattr(self.config.data, 'use_cartesian_state', False) or self.config.data.output_action_dim != 7:
            raise ValueError('DROID evaluation requires Cartesian state and 7D actions')
        self.policy = create_trained_policy(self.config, checkpoint, seed=seed, default_prompt=prompt)

    def sample_actions(self, observation, only_base_actions=True):
        if not only_base_actions:
            raise ValueError('SFT policy has no EXPO critic/edit actor')
        inputs = {(k if k == 'prompt' else 'observation/'+k): v for k, v in observation.items()}
        result = self.policy.infer(inputs)
        return np.asarray(result['actions']), self, result.get('policy_timing', {})


class OnlinePolicy:
    def __init__(self, checkpoint, *, task, initial_sft_checkpoint=None, initial_sft_base=None, seed=42):
        from expo_ft.utils.model_config import read_record, restore_config, check_task
        self.record = read_record(checkpoint)
        config = restore_config(self.record, initial_sft_checkpoint, initial_sft_base)
        check_task(self.record, task, self.record['replan_steps'])
        self.agent = build_agent(config, task, self.record['replan_steps'], seed=seed,
                                 resume=True, num_robot=self.record['num_robot'])
        import orbax.checkpoint as ocp
        from expo_ft.agents.alg.expo_ft import restore_checkpoint
        root = Path(checkpoint).resolve()
        manager = ocp.CheckpointManager(root.parent,
            item_handlers={'agent': ocp.PyTreeCheckpointHandler(), 'params': ocp.PyTreeCheckpointHandler(),
                           'model_config': ocp.JsonCheckpointHandler(filename='config.json')},
            options=ocp.CheckpointManagerOptions(read_only=True))
        try:
            self.agent = restore_checkpoint(manager, self.agent, int(root.name))
        finally:
            manager.close()
        self.agent = self.agent.cache_infer_params()
        self.config = self.agent.actor.train_config
        self.prompt = task.language_instruction

    def sample_actions(self, observation, only_base_actions=False):
        if only_base_actions:
            raise ValueError('Online evaluation uses the saved EXPO action-selection configuration')
        observation = {**observation, 'prompt': observation.get('prompt', self.prompt)}
        actions, self.agent, info = self.agent.sample_actions(observation, only_base_actions=False)
        return actions, self, info


def build_agent(config, task, replan_steps, *, seed=42, resume=False, num_robot=2):
    """Build one-device EXPO graph using replay's schema, without reading a dataset."""
    import jax
    from openpi.training import sharding
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.agents.alg.expo_ft import load_agent
    from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
    from expo_ft.data.replay_buffer import create_replay_buffer, _critic_key_to_storage
    from expo_ft.utils.model_config import make_record
    record = make_record(config, task, replan_steps, num_robot)
    if jax.device_count() != 1:
        raise ValueError('Expose one GPU for model evaluation')
    mesh = sharding.make_mesh(1)
    data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    actor, state, target, kwargs, metadata = build_pi05(config, seed, mesh, data, replicated,
                                                      resume, task.language_instruction)
    actor.checkpoint_record = record
    cameras = tuple(getattr(task, 'critic_camera_keys', CRITIC_CAMERA_KEYS))
    buffer = create_replay_buffer(config=config, example_action=task.example_action, capacity=1,
        task_description=task.language_instruction, replan_steps=replan_steps, seed=seed, delay=0,
        critic_camera_keys=cameras)
    example = {_critic_key_to_storage(k): np.zeros_like(buffer.dataset_dict[_critic_key_to_storage(k)][:1]) for k in cameras}
    example.update(state=np.zeros_like(buffer.dataset_dict['state'][:1]),
                   actions=np.zeros_like(buffer.dataset_dict['actions'][:1]))
    obs, state_example, action = buffer.convert_to_critic_format(example)
    actor.action_dim, actor.state_dim = action.squeeze().shape[-1], state_example.squeeze().shape[-1]
    kwargs.update(critic_camera_keys=cameras, rollout_cache=False)
    return load_agent(seed, obs.squeeze(), action.squeeze(), state_example.squeeze(), actor,
        state, target, kwargs, metadata, mesh, data, replicated, resume, replan_steps,
        task.language_instruction, task.edit_action_xyzg)
