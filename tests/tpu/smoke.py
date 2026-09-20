"""Real EXPO model/update through train_pi_robo; only robot observations are synthetic.

Run from the repository root. Identity normalization and generated demonstrations
are test fixtures, not a policy or dataset suitable for controlling a real robot.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import h5py
import jax
import numpy as np
from openpi.shared import normalize
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train_pi_robo as train
from expo_ft.agents.alg.expo_ft import EXPOLearner


OUTPUT = Path(os.environ["EXPO_SMOKE_OUTPUT"]).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
events = []
errors = []
robots = [dict(step=0, length=16, episodes=0), dict(step=0, length=24, episodes=0)]
updates = 0
stop = threading.Event()


def report(event, **values):
    row = dict(event=event, seconds=round(time.monotonic() - started, 3), **values)
    events.append(row)
    print("[TPU-SMOKE] " + json.dumps(row), flush=True)
    (OUTPUT / "events.json").write_text(json.dumps(events, indent=2))


def observation(robot=0, step=0):
    rng = np.random.default_rng(100 + robot)
    image = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    return dict(exterior_image_1_left=image, exterior_image_2_left=image.copy(),
                wrist_image_left=np.flip(image, axis=1).copy(),
                cartesian_position=np.array([0.4, robot * 0.1, 0.2 + step * 0.001, 0, 0, 0], np.float32),
                gripper_position=np.array([0.5], np.float32), prompt="pick up the cube")


def serve_robot(index):
    try:
        while not stop.is_set():
            try:
                ws = connect(f"ws://127.0.0.1:{18102 + index}", max_size=None,
                             compression=None, open_timeout=5, close_timeout=2)
                break
            except OSError:
                stop.wait(0.2)
        else:
            return
        robot = robots[index]
        with ws:
            for raw in ws:
                request = msgpack_numpy.unpackb(raw)
                op = request["operation"]
                reply = {"status": "success"}
                if op == "create_env":
                    reply.update(env_id=f"synthetic-{index}", task_description="pick up the cube")
                elif op == "reset":
                    robot.update(step=0, episodes=robot["episodes"] + 1, policy_version=updates)
                    reply.update(observation=observation(index), done=False)
                elif op == "step":
                    assert robot["policy_version"] == updates, "policy changed inside an episode"
                    action = np.asarray(request["action"])
                    assert action.shape == (7,) and np.isfinite(action).all()
                    robot["step"] += 1
                    reply.update(action=action, action_type="policy")
                elif op == "get_observation":
                    done = robot["step"] == robot["length"]
                    reply.update(observation=observation(index, robot["step"]), done=done,
                                 success=done, reward=float(done), mask=float(not done))
                else:
                    raise AssertionError(f"unexpected operation {op}")
                ws.send(msgpack_numpy.packb(reply))
    except BaseException:
        errors.append(traceback.format_exc())
        print(errors[-1], flush=True)


def fingerprint(tree):
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        value = np.asarray(jax.device_get(leaf))
        assert np.isfinite(value).all(), "nonfinite parameter"
        digest.update(value.tobytes())
    return digest.hexdigest()


original_sample = EXPOLearner.sample_actions
original_update = EXPOLearner.update
seen_versions = set()


def checked_sample(self, obs, **kwargs):
    assert threading.current_thread() is threading.main_thread()
    assert int(self.actor_train_state.step) == updates
    if updates:
        assert int(self._infer_cache["actor_train_state"].step) == updates
    began = time.monotonic()
    result = original_sample(self, obs, **kwargs)
    actions = np.asarray(jax.device_get(result[0]))
    assert actions.shape == (16, 7) and np.isfinite(actions).all()
    if updates not in seen_versions:
        seen_versions.add(updates)
        report("inference", policy_version=updates, shape=list(actions.shape),
               wall_seconds=round(time.monotonic() - began, 3))
    return result


def checked_update(self, agent, batch, utd_ratio, actor_batch=None):
    global updates
    assert all(r["step"] == r["length"] for r in robots), "update before both episodes ended"
    assert robots[0]["episodes"] == robots[1]["episodes"]
    assert actor_batch is not None, "success-only actor batch missing"
    def hashes(a):
        return dict(actor=fingerprint(a.actor_train_state.params.filter(a.actor.train_config.trainable_filter)),
                    critic=fingerprint(a.critic.params), edit_actor=fingerprint(a.edit_actor.params))
    before = hashes(agent)
    report("update_start", episodes=[r["episodes"] for r in robots],
           critic_batch=list(batch["actions"].shape), actor_batch=list(actor_batch["actions"].shape))
    began = time.monotonic()
    new_agent, info = original_update(self, agent, batch, utd_ratio, actor_batch)
    metrics = {key: float(value) for key, value in jax.device_get(info).items()}
    assert all(np.isfinite(value) for value in metrics.values()), metrics
    after = hashes(new_agent)
    assert all(before[key] != after[key] for key in before), "an expected trainable parameter group did not change"
    updates += 1
    assert int(new_agent.actor_train_state.step) == updates
    assert fingerprint(new_agent._infer_cache["actor_train_state"].params.filter(
        new_agent.actor.train_config.trainable_filter)) == after["actor"], "stale inference parameters"
    report("update_passed", update=updates, changed=list(before), metrics=metrics,
           wall_seconds=round(time.monotonic() - began, 3))
    return new_agent, info


started = time.monotonic()
logging.basicConfig(level=logging.INFO)
assert jax.default_backend() == "tpu", jax.devices()
assert len(os.environ["TPUQ_CHIP_IDS"].split(",")) == 4
report("runtime", jax=jax.__version__, devices=[str(d) for d in jax.devices()],
       base_checkpoint="gs://openpi-assets/checkpoints/pi05_base/params",
       synthetic_data=True)

stats_dir = OUTPUT / "assets" / "synthetic"
normalize.save(stats_dir, {key: normalize.NormStats(mean=np.zeros(7), std=np.ones(7),
    q01=-np.ones(7), q99=np.ones(7)) for key in ("state", "actions")})
demo_dir = OUTPUT / "demo" / "0"
demo_dir.mkdir(parents=True, exist_ok=True)
with h5py.File(demo_dir / "traj.hdf5", "w") as f:
    obs_group = f.create_group("saved_observation")
    frames = [observation(step=t) for t in range(32)]
    for key in frames[0]:
        values = np.asarray([frame[key] for frame in frames])
        if values.dtype.kind == "U":
            values = values.astype("S")
        obs_group.create_dataset(key, data=values)
    action = f.create_group("action")
    action.create_dataset("cartesian_velocity", data=np.full((32, 6), 0.01, np.float32))
    action.create_dataset("gripper_velocity", data=np.zeros(32, np.float32))

train.FLAGS([
    "tpu-smoke", "--num_robot=2", "--update_type=episode", "--num_updates=1",
    "--batch_size=8", "--utd_ratio=1", "--max_steps=320", "--replan_steps=8",
    "--offline_ratio=0.5", "--client_host=127.0.0.1", "--client_port=18102",
    f"--fsdp_devices={jax.device_count()}", "--config=configs/model/expo_ft_pi_config.py",
    "--config.N=2", "--config.n_edit_samples=2", "--config.num_qs=2",
    f"--config.pi05_assets_dir={OUTPUT / 'assets'}", "--config.pi05_asset_id=synthetic",
    "--config_task=configs/task/pick.py", "--config_task.control_hz=1000",
    f"--dataset_path={OUTPUT / 'demo'}", f"--output_dir={OUTPUT / 'training'}",
    "--run_name=tpu-smoke", "--project_name=expo-tpu-smoke", "--overwrite",
    "--checkpoint_buffer", "--checkpoint_model", "--tqdm=false",
])
EXPOLearner.sample_actions = checked_sample
EXPOLearner.update = checked_update
threads = [threading.Thread(target=serve_robot, args=(i,), daemon=True) for i in range(2)]
for thread in threads:
    thread.start()
try:
    train.main(None)
    assert not errors, errors
    assert updates == 3 and seen_versions == {0, 1, 2}, (updates, seen_versions)
    assert [r["episodes"] for r in robots] == [8, 8]
    checkpoint_dir = OUTPUT / "training/tpu-smoke/checkpoints"
    ledger = json.loads((checkpoint_dir / "round-320.json").read_text())
    assert ledger["episode_count"] == 16 and ledger["num_robot"] == 2
    assert [len(list((checkpoint_dir / f"robot-{i}/buffers").glob("*.pkl")))
            for i in range(2)] == [128, 192]
    assert (checkpoint_dir / "320").is_dir()
    report("passed", updates=updates, episodes=16, transitions=320,
           inference_versions=sorted(seen_versions), checkpoint_saved=True)
finally:
    stop.set()
    for thread in threads:
        thread.join(timeout=5)
    (OUTPUT / "errors.json").write_text(json.dumps(errors, indent=2))
