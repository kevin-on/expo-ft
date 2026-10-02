"""One episode per robot, with inference owned by the calling (learner) thread."""
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
import queue
import threading
import time

import numpy as np

from expo_ft.env.model_frame import model_inputs
from expo_ft.env.rollout_timing import step_timing, log_step_timing


def collect_round(envs, sample_actions, replan_steps, control_hz, mirror_robot=None,
                  on_transition=None, on_episode_end=None, check_session=None, *, reset_done=False,
                  wait_for_start=None, canonical_frame=False, mark_handoff=False, round_id=None):
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
    mark_handoff tags the human-to-policy zero poll for split replay filtering;
    it remains in the control stream, including terminal information.
    on_transition receives (robot, step, transition, timing) once per completed
    step. Timing stays separate from the replay record.
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
        if wait_for_start is not None and not wait_for_start(index, stopped):
            return None
        if stopped.is_set():
            return None
        observation = env.start_episode() if reset_done else env.reset()
        observed = time.monotonic()
        policy_observed = None
        if canonical:
            observation = model_inputs(observation)
        plan = deque()
        plan_metadata = None
        transitions = []
        action_type = "policy"
        last_dispatch = None
        while not stopped.is_set():
            plan_ms = 0.
            if not plan and action_type != "human":
                plan_metadata = getattr(env, 'get_observation_metadata', lambda: None)()
                planning = time.monotonic()
                response = Future()
                requests.put((deepcopy(observation), response))
                while not response.done():
                    if stopped.wait(0.01):
                        return None
                if stopped.is_set():
                    return None
                plan.extend(response.result()[:replan_steps])
                plan_ms = (time.monotonic() - planning) * 1000
                policy_observed = observed
            # Observation, RPC and inference time are part of the control period.
            # Anchor each period to the previous actual dispatch, with no catch-up burst.
            if last_dispatch is not None:
                remaining = 1 / control_hz - (time.monotonic() - last_dispatch)
                if stopped.wait(max(0, remaining)):
                    return None
            # While human control is active, poll the override with the existing
            # zero command. Resume inference after a step returns policy control.
            has_action = bool(plan)
            command = plan.popleft() if has_action else np.zeros_like(action)
            last_dispatch = time.monotonic()
            set_metadata = getattr(env, 'set_action_observation', None)
            if set_metadata is not None:
                set_metadata(plan_metadata if has_action else None)
            action, action_type = env.step(command)
            acted = time.monotonic()
            if action_type == "human":
                plan.clear()
            next_observation = env.get_observation()
            arrived = time.monotonic()
            if canonical:
                next_observation = model_inputs(next_observation)
            done, success, reward, mask = env.get_info_for_step()
            timing = step_timing(env, step=len(transitions)+1, source=action_type,
                observed=observed, policy_observed=policy_observed if has_action else None,
                dispatched=last_dispatch, acted=acted, arrived=arrived, plan_ms=plan_ms)
            log_step_timing(timing, mode='online', robot=index, round_id=round_id)
            transitions.append(dict(
                observations=observation, actions=action, rewards=reward,
                masks=mask, dones=done, is_hil=action_type == "human",
            ))
            if mark_handoff and not has_action and action_type != "human":
                transitions[-1]['is_handoff'] = True
            if on_transition is not None:
                on_transition(index, len(transitions) - 1, transitions[-1], timing)
            observation = next_observation
            observed = arrived
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
