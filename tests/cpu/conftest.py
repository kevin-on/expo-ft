"""CPU harness: real replay/JAX/RPC code, stand-ins only for VLA dependencies.

No model weights, tokenizer, ZED SDK, or Polymetis installation is required.
The stand-ins are confined to this test directory's Python process.
"""
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Bypass eager imports of GPU/VLA learners in package __init__ files.
for name, directory in (("expo_ft.agents", "expo_ft/agents"),
                        ("expo_ft.agents.alg", "expo_ft/agents/alg"),
                        ("expo_ft.utils", "expo_ft/utils")):
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT / directory)]
    sys.modules[name] = module
for name in ("openpi", "openpi.training", "openpi.training.config",
             "openpi.training.weight_loaders", "openpi.transforms"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["openpi.training.config"].TrainConfig = object

from expo_ft.data.replay_buffer import PiReplayBuffer, restore_replay_buffer
sys.modules["expo_ft.agents"].restore_replay_buffer = restore_replay_buffer


@pytest.fixture
def make_buffer(monkeypatch):
    def transform(data):
        return dict(data, state=data["state"],
                    image={key: np.full((2, 2, 3), 100, np.uint8) for key in
                           ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
                    image_mask={key: True for key in
                                ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
                    tokenized_prompt=np.zeros(2, int), tokenized_prompt_mask=np.ones(2, bool))
    monkeypatch.setattr(PiReplayBuffer, "_build_transform_pipeline", lambda _: (transform, lambda x: x))

    def create(seed=1, n_step=2):
        config = SimpleNamespace(
            data=SimpleNamespace(create=lambda *_: None), assets_dirs=None,
            model=SimpleNamespace(action_dim=2, action_horizon=4, max_token_len=2),
        )
        buffer = PiReplayBuffer(np.zeros(2), 512, config, resize_size=2, replan_steps=n_step)
        buffer.seed(seed)
        return buffer
    return create


def transition(robot, index, done=False, success=False, hil=False):
    return dict(observations={"state": np.array([robot, index], np.float32)},
                actions=np.array([robot, index], np.float32), rewards=float(done and success),
                masks=float(not done), dones=done, is_success=success, is_hil=hil)
