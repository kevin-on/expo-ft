"""Shared eval/online step telemetry. No model, SDK or hardware dependencies."""
from collections import deque
import json
import logging
import math
import time


def step_timing(env, *, step, source, observed, policy_observed, dispatched,
                acted, arrived, plan_ms):
    """RPC intervals use the caller's monotonic clock; frame ages come from WS.

    observation_ms describes the NEXT observation request. Frame age describes
    the image that produced THIS action, potentially earlier in the same chunk.
    plan_ms includes waiting for the serialized policy, not just GPU inference.
    """
    return dict(step=step, dispatch=dispatched, completed=arrived, action_source=source,
                plan_ms=plan_ms, action_rpc_ms=(acted-dispatched)*1000,
                observation_ms=(arrived-acted)*1000,
                observation_age_ms=(dispatched-observed)*1000,
                policy_observation_age_ms=((dispatched-policy_observed)*1000
                    if source == 'policy' and policy_observed is not None else None),
                action_frame_timing=getattr(env, 'get_action_timing', lambda: {})(),
                observation_breakdown=getattr(env, 'get_observation_timing', lambda: {})())


def log_step_timing(timing, *, mode, robot, episode=None, round_id=None):
    logging.info('[timing][rollout step] %s', json.dumps(dict(
        mode=mode, robot=robot, episode=episode, round=round_id, **timing),
        separators=(',', ':')))


class RolloutMetrics:
    """Last completed action's frame age plus recent completed control-step Hz.

    Owner holds its existing UI lock for writes/reads. Ages are measurements at
    dispatch, so TUI refreshes must not increase them using a different clock.
    """
    def __init__(self):
        self.completed = deque(maxlen=11)  # Ten intervals, including replanning.
        self.frame_age_ms = {}

    def reset(self):
        self.completed.clear()
        self.frame_age_ms = {}

    def step(self, now=None, *, timing=None):
        if timing is not None:
            now = timing['completed']
        self.completed.append(time.monotonic() if now is None else now)
        self.frame_age_ms = {}
        if timing is not None and timing.get('action_source') == 'policy':
            ages = timing.get('action_frame_timing', {}).get('frame_age_ms', {})
            for view in ('side', 'wrist'):
                value = ages.get(view)
                if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                    self.frame_age_ms[view] = value

    def hz(self, now=None):
        if len(self.completed) < 2:
            return None
        now = time.monotonic() if now is None else now
        elapsed = max(now, self.completed[-1]) - self.completed[0]
        # Include waiting for an unfinished step; a stalled RPC lowers live Hz.
        return (len(self.completed) - 1) / elapsed if elapsed > 0 else None

    def snapshot(self, *, active, now=None):
        return dict(rollout_hz=self.hz(now) if active else None,
                    frame_age_ms=dict(self.frame_age_ms) if active else {})


def format_rollout_metrics(state):
    """Identical labels/units in direct eval, receiver eval and online TUI."""
    hz = state.get('rollout_hz')
    hz_text = '—' if hz is None else f'{hz:.1f}'
    ages = state.get('frame_age_ms', {})
    side, wrist = ('—' if ages.get(view) is None else f'{ages[view]:.0f}'
                   for view in ('side', 'wrist'))
    return f'{hz_text} Hz | Frame age S/W: {side}/{wrist} ms'
