import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from openpi_client import msgpack_numpy
from websockets.sync.client import connect
from websockets.asyncio.client import connect as async_connect

from expo_ft.env.env_client import EnvClient, EnvClientWrapper
from expo_ft.utils.robot_round import collect_round
from test_robot_round import FakeEnv


def start_mock_client(client, length):
    client._ensure_server()
    port = client._server.socket.getsockname()[1]
    def serve():
        env = FakeEnv(length)
        packer = msgpack_numpy.Packer()
        with connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            for raw in ws:
                request = msgpack_numpy.unpackb(raw)
                operation = request["operation"]
                reply = {"status": "success"}
                if operation == "create_env":
                    reply.update(env_id="test", task_description="test task")
                elif operation == "reset":
                    reply.update(observation=env.reset(), done=False)
                elif operation == "step":
                    action, kind = env.step(request["action"])
                    reply.update(action=action, action_type=kind)
                elif operation == "get_observation":
                    done, success, reward, mask = env.get_info_for_step()
                    reply.update(observation=env.get_observation(), done=done,
                                 success=success, reward=reward, mask=mask)
                ws.send(packer.pack(reply))
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return thread


def test_two_real_websocket_connections():
    envs = [EnvClientWrapper({}, "127.0.0.1", 0, recover=False, lazy=True) for _ in range(2)]
    clients = [start_mock_client(env.client, length) for env, length in zip(envs, [3, 7])]
    try:
        for version in [1, 2]:
            episodes = collect_round(envs, lambda _: np.full((4, 2), version), 4, 10000)
            assert [len(rows) for rows, _ in episodes] == [3, 7]
            assert all(np.all(row["actions"] == version) for rows, _ in episodes for row in rows)
    finally:
        for env in envs:
            env.close()
    for thread in clients:
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_close_interrupts_wait_for_absent_robot():
    client = EnvClient("127.0.0.1", 0, reconnect=False)
    client._ensure_server()
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(client.create_env, {})
        client.close()
        with pytest.raises(RuntimeError, match="closed"):
            pending.result(timeout=2)


def test_disconnect_does_not_retry_action():
    client = EnvClient("127.0.0.1", 0, reconnect=False)
    client._ensure_server()
    port = client._server.socket.getsockname()[1]
    requests = []
    def disconnect():
        with connect(f"ws://127.0.0.1:{port}") as ws:
            requests.append(msgpack_numpy.unpackb(ws.recv()))
    thread = threading.Thread(target=disconnect, daemon=True)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match="disconnected"):
            client.step("test", np.zeros(2))
        assert len(requests) == 1
    finally:
        client.close()
        thread.join(timeout=2)


@pytest.mark.parametrize("human_steps", [0, 5])
def test_actual_rollout_handler_with_fake_hardware(monkeypatch, human_steps):
    from client import run_client
    instances = []
    class LocalEnv(FakeEnv):
        def __init__(self, **kwargs):
            super().__init__(8)
            instances.append(self)
            self.kwargs = kwargs
        def step(self, action):
            executed, _ = super().step(action)
            return {"executed_action": executed}
    config = run_client.load_task_config("configs/task/pick.py")
    config.env = LocalEnv
    config.bounds = np.asarray(config.bounds, dtype=np.float32)
    overrides = json.loads(json.dumps(dict(robot_server_port=4243,
        bounds=[[0.1, 0.5], [-0.2, 0.2], [0.1, 0.6]], reset_joints=[0.1] * 7,
        randomize_low=[0.0] * 7, randomize_high=[0.2] * 7, camera_serials=["123", "456"])))
    monkeypatch.setattr(run_client, "load_task_config", lambda _: config)
    monkeypatch.setattr(run_client, "_robot_config", overrides)
    monkeypatch.setattr(run_client, "_get_human_override_action", lambda _:
        (np.full(7, 99), True) if instances[0].index < human_steps else (None, False))
    env = EnvClientWrapper({"env_usage": "train"}, "127.0.0.1", 0, recover=False, lazy=True)
    env.client._ensure_server()
    port = env.client._server.socket.getsockname()[1]
    async def dial():
        async with async_connect(f"ws://127.0.0.1:{port}") as ws:
            await run_client._handle_environment_request(ws)
    thread = threading.Thread(target=lambda: asyncio.run(dial()), daemon=True)
    thread.start()
    try:
        calls = []
        def sample(obs):
            calls.append(obs["state"][1])
            return np.ones((8, 7))
        rows, success = collect_round([env], sample, 8, 10000)[0]
        assert len(rows) == 8 and success
        assert instances[0].kwargs["robot_server_port"] == 4243
        for key in ("bounds", "reset_joints", "randomize_low", "randomize_high"):
            actual = instances[0].kwargs[key]
            assert isinstance(actual, np.ndarray)
            np.testing.assert_allclose(actual, overrides[key])
        assert instances[0].kwargs["bounds"].dtype == np.float32
        assert instances[0].kwargs["reset_joints"].dtype == np.float64
        assert instances[0].kwargs["camera_serials"] == ["123", "456"]
        assert calls == ([0, 6] if human_steps else [0])
        assert sum(row["is_hil"] for row in rows) == human_steps
    finally:
        env.close()
        thread.join(timeout=2)
    assert instances[0].closed
    assert not run_client._env_storage
