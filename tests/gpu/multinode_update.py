"""Allocation-only EXPO 4/8-GPU benchmark; never opens robot/client devices.

Run the same committed source, checkpoint and recorded fixture on each host.
The collective phase explicitly checks a reduction spanning every GPU. The
update phase uses an identical global batch on both topologies, excluding
sampling and initial compilation from the steady GPU-update measurement.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def local_value(value):
    """Read a local copy of a fully replicated global JAX value."""
    import jax
    import numpy as np
    if isinstance(value, jax.Array) and not value.is_fully_addressable:
        if not value.is_fully_replicated:
            raise ValueError("Expected a fully replicated value")
        value = value.addressable_shards[0].data
    return np.asarray(jax.device_get(value))


def emit(args, event, **values):
    import jax
    item = dict(event=event, rank=jax.process_index(), time=time.time(), **values)
    print(json.dumps(item), flush=True)
    with (args.output / f"rank-{jax.process_index()}.jsonl").open("a") as stream:
        stream.write(json.dumps(item) + "\n")


def collective(args, mesh):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    data = NamedSharding(mesh, P("batch", None))
    replicated = NamedSharding(mesh, P())
    reduce = jax.jit(lambda value: jnp.mean(value, axis=0),
                     in_shardings=data, out_shardings=replicated)
    measurements = []
    for size_mib in (1, 16, 64):
        count = size_mib * 1024**2 // 4
        first = jax.process_index() * jax.local_device_count()
        host = np.broadcast_to(np.arange(first + 1, first + 1 + jax.local_device_count(),
                                          dtype=np.float32)[:, None],
                               (jax.local_device_count(), count)).copy()
        array = jax.make_array_from_process_local_data(data, host)
        jax.block_until_ready(array)
        for _ in range(2):
            result = reduce(array)
            jax.block_until_ready(result)
        np.testing.assert_array_equal(local_value(result), (jax.device_count() + 1) / 2)
        seconds = []
        for _ in range(5):
            start = time.monotonic()
            result = reduce(array)
            jax.block_until_ready(result)
            seconds.append(time.monotonic() - start)
        measurement = dict(payload_mib_per_gpu=size_mib, seconds=seconds,
                           median_seconds=statistics.median(seconds))
        measurements.append(measurement)
        emit(args, "collective", **measurement)
        del host, array, result
    return measurements


def control(args):
    import hashlib
    import numpy as np
    from expo_ft.distributed.learner_group import LearnerGroup
    group = LearnerGroup()
    for value in (None, False, {'round': 3, 'ready': True}, bytes(range(256)) * 17000):
        def on_leader():
            assert group.leader
            return value
        actual = group.call(on_leader)
        assert actual == value
    expected = np.arange(1500000, dtype=np.float32)
    actual = group.call(lambda: {'data': expected})
    np.testing.assert_array_equal(actual['data'], expected)
    try:
        group.call(lambda: (_ for _ in ()).throw(ValueError('intentional test failure')))
    except (ValueError, RuntimeError) as exc:
        assert 'intentional test failure' in str(exc)
    else:
        raise AssertionError('Leader error was not propagated')
    group.barrier('control-test-done')
    return dict(control_broadcast_exact=True, leader_error_propagated=True,
                bytes_sha256=hashlib.sha256(bytes(range(256)) * 17000).hexdigest())


def update(args, mesh):
    import jax
    import numpy as np
    import expo_ft.agents  # Preserve the agents/data package import order.
    from configs.model.expo_ft_pi_config import get_config
    from configs.task.pick import get_config as get_task
    from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
    from expo_ft.agents.alg.expo_ft import load_agent
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.data.batch_processor import BatchProcessor
    from expo_ft.data.replay_buffer import create_replay_buffer, _critic_key_to_storage
    from jax.sharding import NamedSharding, PartitionSpec as P
    import openpi.training.sharding as sharding

    config, task = get_config(), get_task()
    from config_fixture import configure
    configure(config, args.output / ('config-fixture-'+str(jax.process_index())), args.params, args.assets, args.asset_id)
    manifest = json.loads((args.fixture / "manifest.json").read_text())
    reference = pickle.loads((args.fixture / "reference.pkl").read_bytes())
    data = NamedSharding(mesh, P(sharding.DATA_AXIS))
    replicated = NamedSharding(mesh, P())
    emit(args, "model_initializing")
    actor, actor_state, target, kwargs, metadata = build_pi05(
        config, 42, mesh, data, replicated, False, task.language_instruction)
    # Short, complete real episodes seed the benchmark replay on both hosts.
    episodes = []
    for robot in range(2):
        candidates = manifest["robots"][robot]
        chosen = [row for row in candidates if row["success"]][:2]
        if not chosen:
            raise ValueError("Recorded fixture needs a successful episode for each robot")
        episodes.append([(pickle.loads((args.fixture / row["file"]).read_bytes()),
                          row["success"]) for row in chosen])
    capacity = max(sum(len(rows) for rows, _ in robot) for robot in episodes) + 128
    buffer_args = dict(config=config, example_action=reference["actions"][None],
                       capacity=capacity, task_description=task.language_instruction,
                       replan_steps=8, delay=0, critic_camera_keys=CRITIC_CAMERA_KEYS)
    buffers = [create_replay_buffer(**buffer_args, seed=42 + robot) for robot in range(2)]
    offline = create_replay_buffer(**dict(buffer_args, capacity=1), seed=42)
    for buffer, robot in zip(buffers, episodes):
        for rows, success in robot:
            for row in rows:
                buffer.insert(dict(row, is_success=success))
    # Identical sampling seed/order on both hosts is deliberate for this numerical
    # comparison. Production independent replay sampling is a separate phase.
    processor = BatchProcessor(buffers[0], offline, data, args.batch_size, args.utd_ratio,
                               0, config.actor_success_only, False, replay_buffers=buffers)
    example = {_critic_key_to_storage(k): buffers[0].dataset_dict[_critic_key_to_storage(k)][:1]
               for k in CRITIC_CAMERA_KEYS}
    example.update(state=buffers[0].dataset_dict["state"][:1],
                   actions=buffers[0].dataset_dict["actions"][:1])
    obs, state, action = buffers[0].convert_to_critic_format(example)
    actor.action_dim, actor.state_dim = action.squeeze().shape[-1], state.squeeze().shape[-1]
    kwargs.update(critic_camera_keys=CRITIC_CAMERA_KEYS, rollout_cache=False)
    agent = load_agent(seed=42, example_observation=obs.squeeze(), example_action=action.squeeze(),
                       example_state=state.squeeze(), actor=actor, actor_train_state=actor_state,
                       target_actor_params=target, agent_kwargs=kwargs, metadata=metadata,
                       mesh=mesh, data_sharding=data, replicated_sharding=replicated, resume=False,
                       replan_steps=8, default_prompt=task.language_instruction,
                       edit_action_xyzg=task.edit_action_xyzg)
    batch = processor._sample_buffer(buffers[0], args.batch_size * args.utd_ratio)
    actor_batch = processor._sample_buffer(buffers[0], args.batch_size, success_only=True)
    if actor_batch is None:
        raise ValueError("No eligible successful samples")

    def distribute(tree, blocked=False):
        def leaf(value):
            value = np.asarray(value)
            if blocked:
                value = value.reshape((args.utd_ratio, args.batch_size) + value.shape[1:])
                width = args.batch_size // jax.process_count()
                begin = jax.process_index() * width
                layout = NamedSharding(mesh, P(None, sharding.DATA_AXIS))
                return jax.make_array_from_process_local_data(layout, value[:, begin:begin + width])
            if len(value) % jax.process_count():
                raise ValueError("Batch must divide evenly between hosts")
            width = len(value) // jax.process_count()
            begin = jax.process_index() * width
            return jax.make_array_from_process_local_data(data, value[begin:begin + width])
        return jax.tree.map(leaf, tree)

    batch, actor_batch = distribute(batch, args.utd_axis), distribute(actor_batch)
    jax.block_until_ready((batch, actor_batch, agent))
    emit(args, "fixed_batches_ready", global_batch=args.batch_size, utd=args.utd_ratio,
         batch_layout="UTD/minibatch" if args.utd_axis else "existing flat input")
    seconds = []
    for index in range(2 + args.updates):
        start = time.monotonic()
        agent, info = agent.update(agent, batch, args.utd_ratio, actor_batch)
        jax.block_until_ready((agent, info))
        elapsed = time.monotonic() - start
        values = {name: local_value(value).tolist() for name, value in info.items()}
        if not all(np.isfinite(local_value(value)).all() for value in info.values()):
            raise FloatingPointError("Nonfinite model update")
        emit(args, "update", index=index, compile_or_warmup=index < 2,
             seconds=elapsed, metrics=values)
        if index == 0:
            # Samples from every optimized parameter leaf, outside timed update.
            # Both ranks inspect local replicas, without a parameter all-gather.
            samples = {}
            trees = dict(actor=agent.actor_train_state.params.filter(actor.train_config.trainable_filter),
                         critic=agent.critic.params, encoder=agent.batch_encoder.params,
                         edit=agent.edit_actor.params, temperature=agent.temp.params)
            for path, value in jax.tree_util.tree_flatten_with_path(trees)[0]:
                array = value.addressable_shards[0].data if isinstance(value, jax.Array) else value
                positions = np.linspace(0, array.size - 1, min(array.size, 64), dtype=np.int32)
                samples[jax.tree_util.keystr(path)] = np.asarray(
                    jax.device_get(array.reshape(-1)[positions]), dtype=np.float32)
            np.savez(args.output / f"first-update-samples-rank-{jax.process_index()}.npz", **samples)
        if index >= 2:
            seconds.append(elapsed)
    assert int(local_value(agent.critic.step)) == (2 + args.updates) * args.utd_ratio
    assert int(local_value(agent.actor_train_state.step)) == 2 + args.updates
    return dict(update_seconds=seconds, median_update_seconds=statistics.median(seconds),
                groups_of_three_seconds=[sum(seconds[i:i+3]) for i in range(0, len(seconds)-2, 3)],
                global_batch=args.batch_size, utd=args.utd_ratio,
                input_batch_layout="UTD/minibatch" if args.utd_axis else "existing flat sharding")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("collective", "control", "update"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--asset-id", default="expo_ft/pick_cube_balance_0923_mixed_20_seed3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--utd-ratio", type=int, default=20)
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--utd-axis", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import jax
    processes = int(os.environ.get("EXPO_PROCESS_COUNT", "1"))
    if processes > 1:
        jax.distributed.initialize(coordinator_address=os.environ["EXPO_COORDINATOR"],
                                   num_processes=processes,
                                   process_id=int(os.environ["EXPO_PROCESS_ID"]),
                                   local_device_ids=list(range(4)), initialization_timeout=180)
    import numpy as np
    from jax.sharding import Mesh
    assert jax.local_device_count() == 4
    assert jax.device_count() == processes * 4
    mesh = Mesh(np.asarray(jax.devices()).reshape((-1, 1)), ("batch", "fsdp"))
    emit(args, "devices", jax_version=jax.__version__, global_devices=jax.device_count(),
         devices=[str(device) for device in jax.devices()])
    result = (collective(args, mesh) if args.phase == "collective" else
              control(args) if args.phase == "control" else update(args, mesh))
    (args.output / f"result-rank-{jax.process_index()}.json").write_text(json.dumps(result, indent=2) + "\n")
    emit(args, "passed", phase=args.phase)
    if processes > 1:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
