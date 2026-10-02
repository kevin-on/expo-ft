"""Recent completed control-step rate, independent of TUI refresh frequency."""
from collections import deque
import time


class RolloutRate:
    def __init__(self):
        self.completed = deque(maxlen=11)  # Ten intervals, including replanning.

    def reset(self):
        self.completed.clear()

    def step(self, now=None):
        self.completed.append(time.monotonic() if now is None else now)

    def hz(self, now=None):
        if len(self.completed) < 2:
            return None
        now = time.monotonic() if now is None else now
        elapsed = max(now, self.completed[-1]) - self.completed[0]
        # Include waiting for an unfinished step, so a stalled RPC doesn't keep
        # displaying the last healthy rate. Caller hides this while not rolling.
        return (len(self.completed) - 1) / elapsed if elapsed > 0 else None
