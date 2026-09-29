"""Hardware-free real-model checks on an allocated GPU. Never opens robot clients.

Old SFT metadata is wrapped as an explicit TEST fixture; original artifacts are read-only.
Run sft, train, restore in separate processes to avoid retaining duplicate model buffers.
"""
import argparse
import json
from pathlib import Path
import shutil
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['sft','train','restore'], required=True)
    p.add_argument('--sft', type=Path, required=True)
    p.add_argument('--demo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import jax
    import numpy as np
    from configs.task.pick import get_config as get_task
    from configs.model.expo_ft_pi_config import get_config
    from expo_ft.env.droid_utils import process_droid_dataset
    from expo_ft.env.checkpoint_policy import SFTPolicy, OnlinePolicy, build_agent
    from openpi.training import checkpoint_config as recipe, config as op
    task=get_task()
    assert jax.default_backend()=='gpu'
    dataset=process_droid_dataset(str(a.demo),task,num_data=1)
    assert dataset
    obs=dataset[0]['observations']
    fixture=a.output/'sft'
    if not fixture.exists():
        fixture.mkdir()
        (fixture/'params').symlink_to(a.sft.resolve()/'params',target_is_directory=True)
        shutil.copytree(a.sft/'assets',fixture/'assets')
        original=json.loads((fixture/'assets/config.json').read_text())
        if original['version']==1:
            # This validation fixture explicitly uses the historical CLI plus today's pinned preset.
            # It is not a migration of or a write to the original trained checkpoint.
            rec=recipe.make_record(op.cli(original['config_args']),original['config_args'])
            (fixture/'assets/config.json').write_text(json.dumps(rec))
    began=time.monotonic()
    def report(event,**kw):
        row=dict(event=event,seconds=time.monotonic()-began,**kw)
        print(json.dumps(row),flush=True)
        with (a.output/(a.phase+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    if a.phase=='sft':
        policy=SFTPolicy(fixture,prompt=task.language_instruction)
        actions,_,_=policy.sample_actions(obs)
        assert actions.shape==(16,7) and np.isfinite(actions).all()
        report('SFT_INFERENCE_OK',omit=list(policy.config.model.omit_image_keys))
        return
    if a.phase=='restore':
        policy=OnlinePolicy(a.output/'checkpoints/1',task=task)
        expected=json.loads((a.output/'expected.json').read_text())
        assert policy.agent.N==expected['N'] and policy.agent.n_edit_samples==expected['edits']
        assert int(policy.agent.actor_train_state.step)==1
        actions,_,_=policy.sample_actions(obs)
        np.testing.assert_allclose(np.asarray(actions),np.load(a.output/'expected.npy'),rtol=1e-4,atol=1e-5)
        report('ONLINE_RESTORE_PARITY_OK',N=policy.agent.N,actor_step=int(policy.agent.actor_train_state.step))
        return
    config=get_config();config.initial_sft_checkpoint=str(fixture)
    # Nondefault config proves the restored policy does not quietly use default 8/8.
    config.N=2;config.n_edit_samples=2
    agent=build_agent(config,task,8)
    report('AGENT_READY')
    actions,agent,_=agent.sample_actions(obs)
    assert np.asarray(actions).shape==(16,7) and np.isfinite(np.asarray(actions)).all()
    from expo_ft.data.replay_buffer import create_replay_buffer
    from expo_ft.data.batch_processor import BatchProcessor
    from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
    from openpi.training import sharding
    mesh=sharding.make_mesh(1)
    data_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    def buf():return create_replay_buffer(config,task.example_action,len(dataset)+16,
        task.language_instruction,8,42,critic_camera_keys=CRITIC_CAMERA_KEYS)
    processor=BatchProcessor(replay_buffer=buf(),offline_replay_buffer=buf(),data_sharding=data_sharding,
        batch_size=2,utd_ratio=1,offline_ratio=0,actor_success_only=config.actor_success_only,
        use_dagger_hil_sampling=False,dataset=dataset)
    batch,actor_batch,_=processor.next_batch(jax.random.PRNGKey(2))
    report('UPDATE_BEGIN')
    agent,info=agent.update(agent,batch,1,actor_batch)
    jax.block_until_ready((agent,info))
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(info))
    assert int(agent.actor_train_state.step)==1
    from expo_ft.agents import initialize_checkpoint_dir
    from expo_ft.agents.alg.expo_ft import save_checkpoint
    manager,_=initialize_checkpoint_dir(a.output/'checkpoints',keep_period=None,overwrite=False,resume=False)
    save_checkpoint(manager,agent,1);manager.wait_until_finished();manager.close()
    report('UPDATE_CHECKPOINT_OK')
    actions,_,_=agent.cache_infer_params().sample_actions(obs)
    np.save(a.output/'expected.npy',np.asarray(actions))
    (a.output/'expected.json').write_text(json.dumps(dict(N=2,edits=2)))
    report('EXPECTED_ACTION_SAVED')

if __name__=='__main__':main()
