import threading
import time

import numpy as np
import pytest

from expo_ft.utils.robot_round import collect_round, updates_for_round


class FakeEnv:
    def __init__(self, length, robot=0, human_at=None, fail_at=None):
        self.length, self.robot = length, robot
        self.human_at, self.fail_at = human_at, fail_at
        self.closed = False
        self.episodes = []

    def reset(self):
        self.index = 0
        self.actions = []
        self.episodes.append(self.actions)
        return self.get_observation()

    def step(self, action):
        assert self.index < self.length, "post-terminal action"
        if self.index == self.fail_at:
            raise RuntimeError("robot disconnected")
        action = np.asarray(action).copy()
        human = self.index == self.human_at
        if human:
            action[:] = 99
        self.actions.append(action)
        self.index += 1
        return action, "human" if human else "policy"

    def get_observation(self):
        return {"state": np.array([self.robot, self.index])}

    def get_info_for_step(self):
        done = self.index == self.length
        return done, done, float(done), float(not done)

    def close(self):
        self.closed = True


@pytest.mark.parametrize("num_robot", [1, 2, 3])
def test_barrier_policy_version_and_transition_alignment(num_robot):
    envs = [FakeEnv(2 + 3 * i, robot=i) for i in range(num_robot)]
    caller = threading.get_ident()
    for version in range(2):
        def sample(obs):
            assert threading.get_ident() == caller
            obs["state"][:] = -123  # VLA preprocessing must not mutate stored observations
            return np.full((4, 2), version)
        episodes = collect_round(envs, sample, 2, 10000)
        assert len(episodes) == num_robot
        for env, (rows, success) in zip(envs, episodes):
            assert env.index == env.length and success
            assert len(env.episodes) == version + 1
            assert len(rows) == env.length
            assert rows[-1]["dones"] and rows[-1]["rewards"] == 1
            assert not any(row["dones"] for row in rows[:-1])
            for index, row in enumerate(rows):
                np.testing.assert_array_equal(row["observations"]["state"], [env.robot, index])
                assert np.all(row["actions"] == version)


def test_human_override_discards_remaining_chunk():
    env = FakeEnv(4, human_at=1)
    calls = []
    def sample(obs):
        calls.append(obs["state"][1])
        return np.full((8, 2), len(calls))
    rows, _ = collect_round([env], sample, 8, 10000)[0]
    assert calls == [0, 3]
    assert rows[1]["is_hil"] and np.all(rows[1]["actions"] == 99)
    assert np.all(rows[2]["actions"] == 0)  # zero-command handoff, as in the original loop
    assert np.all(rows[3]["actions"] == 2)


def test_continuous_human_control_skips_inference_and_resumes():
    class HumanEnv(FakeEnv):
        def step(self, action):
            self.human_at = self.index if self.index < 5 else None
            return super().step(action)
    envs = [HumanEnv(8, robot=0), FakeEnv(9, robot=1)]
    calls = [[], []]
    def sample(obs):
        robot, index = obs["state"]
        calls[robot].append(index)
        return np.full((8, 2), len(calls[robot]))
    human_rows, _ = collect_round(envs, sample, 8, 10000)[0]
    assert calls[0] == [0, 6]  # no inference during all five human steps or handoff
    assert calls[1] == [0, 8]
    assert all(row["is_hil"] and np.all(row["actions"] == 99) for row in human_rows[:5])
    assert not human_rows[5]["is_hil"] and np.all(human_rows[5]["actions"] == 0)
    assert np.all(human_rows[6]["actions"] == 2)
    assert human_rows[-1]["dones"]


def test_human_control_through_episode_end_and_next_round():
    class HumanEnv(FakeEnv):
        def step(self, action):
            self.human_at = self.index
            return super().step(action)
    env = HumanEnv(5)
    calls = []
    def sample(obs):
        calls.append(obs["state"][1])
        return np.ones((4, 2))
    for _ in range(2):
        rows, success = collect_round([env], sample, 4, 10000)[0]
        assert len(rows) == 5 and success and rows[-1]["dones"]
        assert all(row["is_hil"] for row in rows)
    assert calls == [0, 0]


def test_worker_failure_stops_round():
    envs = [FakeEnv(10000), FakeEnv(5, fail_at=1)]
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="disconnected"):
        collect_round(envs, lambda _: np.ones((2, 2)), 2, 10000)
    assert all(env.closed for env in envs)
    assert time.monotonic() - started < 2


def test_inference_failure_stops_waiting_workers():
    envs = [FakeEnv(10), FakeEnv(10)]
    def fail(_):
        raise ValueError("inference failed")
    with pytest.raises(ValueError, match="inference"):
        collect_round(envs, fail, 2, 10000)
    assert all(env.closed for env in envs)


def test_update_budget():
    common = dict(num_updates=0, step_interval=8)
    assert updates_for_round(0, 19, can_update=False, **common) == (0, 0)
    assert updates_for_round(0, 19, can_update=True, **common) == (2, 3)
    assert updates_for_round(3, 13, can_update=True, **common) == (2, 0)
    assert updates_for_round(0, 19, can_update=True, num_updates=3, step_interval=8) == (3, 0)


@pytest.mark.parametrize("replan_steps,inference_delay,expected_period", [
    (8, 0.0, 0.1), (1, 0.02, 0.1), (1, 0.12, 0.17),
])
def test_dispatch_period_includes_rpc_observation_and_inference(replan_steps, inference_delay, expected_period):
    class DelayedEnv(FakeEnv):
        def reset(self):
            self.dispatches = []
            return super().reset()
        def step(self, action):
            self.dispatches.append(time.monotonic())
            time.sleep(0.01)  # step RPC latency
            return super().step(action)
        def get_observation(self):
            time.sleep(0.04)
            return super().get_observation()
    env = DelayedEnv(8)
    def sample(_):
        time.sleep(inference_delay)
        return np.ones((8, 2))
    rows, success = collect_round([env], sample, replan_steps, 10)[0]
    intervals = np.diff(env.dispatches)
    assert len(rows) == 8 and success
    assert np.all(intervals >= expected_period - 0.005)
    assert np.median(intervals) < expected_period + 0.025


def test_each_robot_keeps_its_own_dispatch_period():
    class DelayedEnv(FakeEnv):
        def __init__(self, delay):
            super().__init__(6)
            self.delay = delay
            self.dispatches = []
        def step(self, action):
            self.dispatches.append(time.monotonic())
            return super().step(action)
        def get_observation(self):
            time.sleep(self.delay)
            return super().get_observation()
    envs = [DelayedEnv(0.02), DelayedEnv(0.04)]
    collect_round(envs, lambda _: np.ones((8, 2)), 8, 10)
    for env in envs:
        intervals = np.diff(env.dispatches)
        assert np.all(intervals >= 0.095)
        assert np.median(intervals) < 0.12
