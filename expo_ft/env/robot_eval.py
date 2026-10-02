"""Concurrent robot evaluation around one serialized policy; hardware is injected."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time

import numpy as np
from expo_ft.env.model_frame import model_inputs
from expo_ft.env.rollout_timing import RolloutMetrics, step_timing, log_step_timing


class RobotEvaluation:
    def __init__(self, envs, predict, *, replan_steps, control_hz, max_steps):
        if not envs or not set(envs) <= {0, 1}:
            raise ValueError('Select robot 0, 1, or both')
        if min(replan_steps, control_hz, max_steps) <= 0:
            raise ValueError('Positive replan/control frequency/episode length required')
        self.envs, self.predict = envs, predict
        self.replan_steps, self.period, self.max_steps = replan_steps, 1/control_hz, max_steps
        self.stop = threading.Event()
        self.lock, self.policy_lock = threading.Lock(), threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=len(envs), thread_name_prefix='robot-eval')
        self.pending, self.barrier = [], None
        self.by_robot, self.barriers = {}, []
        self.results = queue.SimpleQueue()
        self.states = {r: dict(status='waiting', steps=0, episodes=0, successes=0, last=None) for r in envs}
        self.metrics = {r: RolloutMetrics() for r in envs}

    def snapshot(self):
        with self.lock:
            return {r: dict(s, **self.metrics[r].snapshot(active=s['status']=='running'))
                    for r,s in self.states.items()}

    def state(self, robot, *, timing=None, **values):
        with self.lock:
            if values.get('status') == 'starting':
                self.metrics[robot].reset()
            elif values.get('status') == 'running' and 'steps' in values:
                self.metrics[robot].step(timing=timing)
            self.states[robot].update(values)

    def check(self):
        if self.stop.is_set():
            raise RuntimeError('Evaluation stopped')

    def fail(self, robot):
        self.state(robot, status='error')
        self.stop.set()
        for barrier in self.barriers:
            barrier.abort()

    def reset(self, robot):
        try:
            self.check()
            self.state(robot, status='resetting')
            self.envs[robot].reset_only()
            self.check()
            self.state(robot, status='ready')
        except BaseException:
            self.fail(robot)
            raise

    def prepare(self):
        if self.pending:
            raise RuntimeError('Already prepared')
        self.by_robot = {r: self.pool.submit(self.reset, r) for r in self.envs}
        self.pending = list(self.by_robot.values())

    def poll(self):
        # Surface a worker failure even if another client is still connecting.
        for future in self.pending:
            if future.done():
                future.result()
        return all(f.done() for f in self.pending) and all(s['status']=='ready' for s in self.snapshot().values())

    def start(self, number, robots=None):
        self.check()
        self.poll()  # Surface errors, even when only one robot is selected.
        robots = list(self.envs if robots is None else robots)
        if not robots or len(set(robots)) != len(robots) or not set(robots) <= self.envs.keys():
            return False
        states = self.snapshot()
        if any(states[r]['status'] != 'ready' or r not in self.by_robot or not self.by_robot[r].done() for r in robots):
            return False
        barrier = threading.Barrier(len(robots))
        self.barrier = barrier
        self.barriers = [b for b in self.barriers if b.n_waiting] + [barrier]
        for robot in robots:
            self.state(robot, status='starting', steps=0)
        for robot in robots:
            episode = number[robot] if isinstance(number, dict) else number
            self.by_robot[robot] = self.pool.submit(self.episode, robot, episode, barrier)
        self.pending = list(self.by_robot.values())
        return True

    def reset_ready(self, robot):
        """Repeat a selected ready robot's reset without recording an episode."""
        self.check()
        self.poll()
        with self.lock:
            if (robot not in self.states or self.states[robot]['status'] != 'ready'
                    or robot not in self.by_robot or not self.by_robot[robot].done()):
                return False
            self.states[robot]['status'] = 'resetting'
            self.by_robot[robot] = self.pool.submit(self.reset, robot)
            self.pending = list(self.by_robot.values())
        return True

    def plan(self, robot, observation):
        observation = model_inputs(observation)
        began = time.monotonic()
        with self.policy_lock:
            self.check()
            actions = np.asarray(self.predict(observation))
        self.check()
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < self.replan_steps or not np.isfinite(actions).all():
            raise ValueError('Policy returned an invalid action chunk')
        return deque(actions[:self.replan_steps]), (time.monotonic()-began)*1000

    def episode(self, robot, number, barrier):
        env = self.envs[robot]
        try:
            self.check()
            observation = env.start_episode()  # Fresh after Space, not a frame from reset.
            observed = time.monotonic()
            plan_metadata = getattr(env, "get_observation_metadata", lambda: None)()
            plan, plan_ms = self.plan(robot, observation)
            policy_observed = observed
            self.state(robot, status='first action ready')
            barrier.wait()  # Neither arm starts while the other first plan is compiling.
            self.check()
            began = time.monotonic()
            previous = None
            human_steps, episode_return, timings = 0, 0., []
            for step in range(1, self.max_steps+1):
                self.check()
                if not plan:
                    plan_metadata = getattr(env, "get_observation_metadata", lambda: None)()
                    plan, plan_ms = self.plan(robot, observation)
                    policy_observed = observed
                if previous is not None:
                    self.stop.wait(max(0., self.period-(time.monotonic()-previous)))
                self.check()
                dispatched = time.monotonic()
                set_metadata = getattr(env, 'set_action_observation', None)
                if set_metadata is not None:
                    set_metadata(plan_metadata)
                _, source = env.step(plan.popleft())
                acted = time.monotonic()
                human_steps += source == 'human'
                observation = env.get_observation()
                arrived = time.monotonic()
                done, success, reward, _ = env.get_info_for_step()
                episode_return += float(reward)
                timing = step_timing(env, step=step, source=source, observed=observed,
                    policy_observed=policy_observed, dispatched=dispatched, acted=acted,
                    arrived=arrived, plan_ms=plan_ms)
                timings.append(timing)
                log_step_timing(timing, mode='eval', robot=robot, episode=number)
                self.state(robot, status='running', steps=step, timing=timing)
                previous, observed, plan_ms = dispatched, arrived, 0.
                if done:
                    break
            else:
                raise RuntimeError(f'Robot {robot} did not terminate at configured max_steps={self.max_steps}')
            row = dict(robot=robot, episode=number, success=bool(success), steps=step,
                human_steps=int(human_steps), intervention_rate=human_steps/step,
                had_intervention=bool(human_steps), success_without_intervention=bool(success and not human_steps),
                episode_return=episode_return, seconds=time.monotonic()-began, timings=timings)
            with self.lock:
                state = self.states[robot]
                state.update(episodes=state['episodes']+1, successes=state['successes']+int(success), last=bool(success))
            self.results.put(row)
            self.reset(robot)  # Each finished robot resets independently; next round still needs Space.
            return row
        except BaseException:
            self.fail(robot)
            raise

    def close(self):
        self.stop.set()
        for barrier in self.barriers:
            barrier.abort()
        try:
            for env in self.envs.values():
                env.close()  # Unblocks accept/RPC; recover=False prevents recreation/reset.
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)
