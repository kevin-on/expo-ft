"""Canonical-frame observation/action transforms matching the mixed SFT dataset."""

import numpy as np


_IMAGE_KEYS = ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")


def canonical_observation(observation, mirror):
    """Transform selected RGB views and raw Cartesian pose BEFORE normalization.

    The WS config selects side right/wrist left for robot0, and side left/wrist
    right for robot1. Keys retain the legacy '*_left' names in both cases.
    Only model inputs are returned; physical camera calibration is not mirrored.
    """
    result = {key: observation[key] for key in (*_IMAGE_KEYS, "gripper_position")}
    pose = np.array(observation["cartesian_position"], dtype=np.float32, copy=True)
    if pose.shape != (6,):
        raise ValueError(f"Expected a six-component Cartesian pose, got {pose.shape}")
    if mirror:
        pose[[1, 3, 5]] *= -1
        for key in _IMAGE_KEYS:
            result[key] = np.ascontiguousarray(np.asarray(result[key])[:, ::-1, :])
    result["cartesian_position"] = pose
    if "prompt" in observation:
        result["prompt"] = observation["prompt"]
    return result


def physical_action(action, mirror):
    """Map unnormalized 7D policy actions back to the physical robot frame."""
    result = np.array(action, dtype=np.float64, copy=True)
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError("SFT policy must return a finite 7D Cartesian-velocity action")
    if mirror:
        result[[1, 3, 5]] *= -1
    return result
