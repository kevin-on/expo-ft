"""Terminal-only rollout controls; no model, robot or network dependencies."""
import logging
import os
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty


class RolloutDashboard:
    def __init__(self, num_robot, max_steps, mode="manual"):
        self.mode, self.max_steps = mode, max_steps
        self.condition = threading.Condition()
        self.closed = threading.Event()
        self.quit_requested = False
        self.round = 0
        self.policy = "-"
        self.phase = "Waiting for learner policy"
        self.message = ""
        self.states = [dict(status="connecting", steps=0, completed=0, successes=0,
                            last="-", seconds=0., started=None, human=0) for _ in range(num_robot)]
        self.fd = None
        self.thread = None
        self.lines = 0
        self.log_handler = None
        self.old_handlers = None
        self.escape = None

    def check(self):
        if self.quit_requested:
            raise RuntimeError("Rollout stopped from inference dashboard")

    def set_phase(self, phase):
        with self.condition:
            self.phase = phase

    def resetting(self, robot):
        with self.condition:
            self.states[robot].update(status="resetting", started=None)

    def reset_done(self, robot):
        with self.condition:
            self.states[robot]["status"] = "waiting policy"

    def ready(self, round_id, policy):
        with self.condition:
            # Never carry start keys typed during reset/update into a new round.
            if self.fd is not None:
                termios.tcflush(self.fd, termios.TCIFLUSH)
            self.round, self.policy = round_id + 1, policy
            self.phase = "Rollout"
            for state in self.states:
                state.update(status="ready", steps=0, seconds=0., started=None, human=0)
            self.condition.notify_all()

    def handle_key(self, key):
        with self.condition:
            if key == b"m":
                self.mode = "manual" if self.mode == "auto" else "auto"
                self.message = f"Mode: {self.mode}; running episodes continue"
            elif key == b"q":
                self.quit_requested = True
            elif key in (b"r", b"t"):
                robot = 0 if key == b"r" else 1
                if self.mode == "manual" and robot < len(self.states) and self.states[robot]["status"] == "ready":
                    self.states[robot]["status"] = "resetting"
                    self.message = f"Reset robot {robot}; press start after READY"
                else:
                    self.message = "Reset ignored: requires a READY robot in manual mode"
            elif key in (b"0", b"1", b" "):
                robots = range(len(self.states)) if key == b" " else [int(key)]
                started = []
                for robot in robots:
                    if robot < len(self.states) and self.states[robot]["status"] == "ready":
                        self.states[robot]["status"] = "starting"
                        started.append(str(robot))
                self.message = "Start: " + ", ".join(started) if started else "No READY robot; start ignored"
            self.condition.notify_all()

    def wait_for_start(self, robot, stopped, reset=None):
        while not stopped.is_set() and not self.quit_requested:
            with self.condition:
                state = self.states[robot]
                if state["status"] == "starting" or (state["status"] == "ready" and self.mode == "auto"):
                    state.update(status="starting", started=time.monotonic())
                    return True
                if state["status"] != "resetting":
                    self.condition.wait(.1)
                    continue
            # This robot's collector owns its RPCs; never hold the UI lock during
            # motion or allow a start request to race the reset.
            try:
                if reset is None:
                    raise RuntimeError("Manual reset callback is unavailable")
                reset()
            except BaseException:
                with self.condition:
                    self.states[robot]["status"] = "error"
                raise
            with self.condition:
                if stopped.is_set() or self.quit_requested:
                    return False
                self.states[robot]["status"] = "ready"
                self.condition.notify_all()
        return False

    def step(self, robot, steps, is_hil):
        with self.condition:
            state = self.states[robot]
            state.update(status="human" if is_hil else "running", steps=steps)
            state["human"] += int(bool(is_hil))

    def episode_done(self, robot, success):
        with self.condition:
            state = self.states[robot]
            state["seconds"] = time.monotonic() - state["started"]
            state.update(status="done", last="success" if success else "failure", started=None)
            state["completed"] += 1
            state["successes"] += int(bool(success))

    def draw(self):
        with self.condition:
            rows = [f"Online FT | {self.mode.upper()} | Round {self.round} | Policy {self.policy} | {self.phase}",
                    "0/1: start robot  Space: start READY robots  r/t: reset robot0/1 (manual READY)  m: auto/manual  q: stop",
                    "Robot  Status          Step   Episodes   Success       Last       Seconds  Human"]
            for robot, state in enumerate(self.states):
                count = state["completed"]
                rate = state["successes"] / count if count else 0.
                seconds = time.monotonic() - state["started"] if state["started"] is not None else state["seconds"]
                rows.append(f"  {robot}    {state['status']:<14} {state['steps']:>3}/{self.max_steps:<3}"
                            f"   {count:>4}     {state['successes']:>3} ({rate:>4.0%})"
                            f"  {state['last']:<9} {seconds:>6.1f}s  {state['human']:>4}")
            rows.append(self.message)
        if self.lines:
            print(f"\033[{self.lines}F", end="")
        print("\033[?25l" + "\n".join("\033[2K" + row for row in rows), flush=True)
        self.lines = len(rows)

    def _terminal_loop(self):
        try:
            while not self.closed.is_set():
                # Serialize reads with ready() so pre-ready input is never queued.
                with self.condition:
                    if select.select([self.fd], [], [], 0)[0]:
                        keys = os.read(self.fd, 64)
                        if not keys:
                            self.quit_requested = True
                        else:
                            for key in keys:
                                # An arrow/function key may arrive across reads;
                                # never interpret its numeric CSI bytes as starts.
                                if self.escape == "prefix":
                                    self.escape = "sequence" if key in (ord('['), ord('O')) else None
                                    continue
                                if self.escape == "sequence":
                                    if 0x40 <= key <= 0x7e:
                                        self.escape = None
                                    continue
                                if key == 0x1b:
                                    self.escape = "prefix"
                                    continue
                                self.handle_key(bytes([key]))
                self.draw()
                self.closed.wait(.1)
        except BaseException:
            logging.exception("Inference dashboard failed")
            self.quit_requested = True

    def start(self, log_path):
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise ValueError("Rollout dashboard requires a terminal: use ssh -tt and srun --pty")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log_handler = logging.FileHandler(log_path)
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.old_handlers = logging.getLogger().handlers[:]
        logging.getLogger().handlers = [self.log_handler]
        self.fd = sys.stdin.fileno()
        self.old_termios = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.message = f"Log: {log_path} | Episode totals are for this inference session"
        self.thread = threading.Thread(target=self._terminal_loop, name="rollout-dashboard", daemon=True)
        self.thread.start()

    def close(self):
        self.closed.set()
        if self.thread is not None:
            self.thread.join()
            self.draw()
        if self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_termios)
            print("\033[?25h", end="", flush=True)
        if self.old_handlers is not None:
            logging.getLogger().handlers = self.old_handlers
        if self.log_handler is not None:
            self.log_handler.close()
