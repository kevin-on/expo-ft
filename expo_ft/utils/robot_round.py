"""One episode per robot, with inference owned by the calling (learner) thread."""
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
import queue
import threading
import time

import numpy as np

from expo_ft.env.sft_eval import canonical_observation, physical_action


def collect_round(envs, sample_actions, replan_steps, control_hz, mirror_robot=None,
                  on_transition=None, on_episode_end=None, check_session=None, *, reset_done=False,
                  wait_for_start=None, canonical_frame=False):
    """Return episodes in robot order. No reset or inference survives this barrier.

    Workers perform RPCs concurrently and submit inference requests to one queue.
    The caller alone touches the policy, including its RNG. On failure, close all
    connections to interrupt pending RPCs, and discard the incomplete round.
    canonical_frame=True also normalizes unmirrored single-robot observations
    and validates executed Cartesian actions, just as for robot0 in a paired round.
    An optional session check also runs while workers wait for WS/reset replies.
    With reset_done=True the caller has joined deferred resets; capture a fresh
    first observation without resetting again or running termination detection.
    An optional start gate runs before reading that observation, independently
    for each robot. Waiting workers do not block the other robot's inference.
    """
    requests = queue.Queue()
    stopped = threading.Event()
    next_check = 0.0

    def check(force=False):
        nonlocal next_check
        if check_session is not None and (force or time.monotonic() >= next_check):
            check_session()
            next_check = time.monotonic() + 0.1

    def collect(index, env):
        if stopped.is_set():
            return None
        canonical = canonical_frame or mirror_robot is not None
        mirror = index == mirror_robot
        if wait_for_start is not None and not wait_for_start(index, stopped):
            return None
        if stopped.is_set():
            return None
        observation = env.start_episode() if reset_done else env.reset()
        if canonical:
            observation = canonical_observation(observation, mirror)
        plan = deque()
        transitions = []
        action_type = "policy"
        last_dispatch = None
        while not stopped.is_set():
            if not plan and action_type != "human":
                response = Future()
                requests.put((deepcopy(observation), response))
                while not response.done():
                    if stopped.wait(0.01):
                        return None
                if stopped.is_set():
                    return None
                plan.extend(response.result()[:replan_steps])
            # Observation, RPC and inference time are part of the control period.
            # Anchor each period to the previous actual dispatch, with no catch-up burst.
            if last_dispatch is not None:
                remaining = 1 / control_hz - (time.monotonic() - last_dispatch)
                if stopped.wait(max(0, remaining)):
                    return None
            # While human control is active, poll the override with the existing
            # zero command. Resume inference after a step returns policy control.
            command = plan.popleft() if plan else np.zeros_like(action)
            last_dispatch = time.monotonic()
            if canonical:
                command = physical_action(command, mirror)
            action, action_type = env.step(command)
            if canonical:
                # Reflection is its own inverse. Store the executed action,
                # including workspace clipping and human overrides, in the model frame.
                action = physical_action(action, mirror)
            if action_type == "human":
                plan.clear()
            next_observation = env.get_observation()
            if canonical:
                next_observation = canonical_observation(next_observation, mirror)
            done, success, reward, mask = env.get_info_for_step()
            transitions.append(dict(
                observations=observation, actions=action, rewards=reward,
                masks=mask, dones=done, is_hil=action_type == "human",
            ))
            if on_transition is not None:
                on_transition(index, len(transitions) - 1, transitions[-1])
            observation = next_observation
            if done:
                if on_episode_end is not None:
                    on_episode_end(index, len(transitions), success)
                return transitions, success

    with ThreadPoolExecutor(max_workers=len(envs)) as workers:
        try:
            check(force=True)
            episodes = [workers.submit(collect, index, env) for index, env in enumerate(envs)]
            while not all(episode.done() for episode in episodes):
                check()
                for episode in episodes:
                    if episode.done():
                        episode.result()  # surface failures before another policy request
                try:
                    observation, response = requests.get(timeout=0.01)
                except queue.Empty:
                    continue
                actions = np.asarray(sample_actions(observation))
                check(force=True)  # sampling/JIT may have blocked since the last check
                response.set_result(actions)
            check(force=True)
            return [episode.result() for episode in episodes]
        except BaseException:
            stopped.set()
            for env in envs:
                env.close()
            raise


def updates_for_round(pending_steps, round_steps, *, can_update, num_updates, step_interval):
    """Keep the existing warmup and fixed-count / transition-count update rules."""
    if not can_update:
        return 0, 0
    if num_updates > 0:
        return num_updates, 0
    return divmod(pending_steps + round_steps, step_interval)
