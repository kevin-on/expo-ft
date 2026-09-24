"""One episode per robot, with inference owned by the calling (learner) thread."""
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
import queue
import threading
import time

import numpy as np

from expo_ft.env.sft_eval import canonical_observation, physical_action


def collect_round(envs, sample_actions, replan_steps, control_hz, mirror_robot=None):
    """Return episodes in robot order. No reset or inference survives this barrier.

    Workers perform RPCs concurrently and submit inference requests to one queue.
    The caller alone touches the policy, including its RNG. On failure, close all
    connections to interrupt pending RPCs, and discard the incomplete round.
    """
    requests = queue.Queue()
    stopped = threading.Event()

    def collect(index, env):
        canonical = mirror_robot is not None
        mirror = index == mirror_robot
        observation = env.reset()
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
            observation = next_observation
            if done:
                return transitions, success

    with ThreadPoolExecutor(max_workers=len(envs)) as workers:
        episodes = [workers.submit(collect, index, env) for index, env in enumerate(envs)]
        try:
            while not all(episode.done() for episode in episodes):
                for episode in episodes:
                    if episode.done():
                        episode.result()  # surface failures before another policy request
                try:
                    observation, response = requests.get(timeout=0.01)
                except queue.Empty:
                    continue
                response.set_result(np.asarray(sample_actions(observation)))
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
