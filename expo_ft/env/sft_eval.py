"""Canonical-frame observation/action transforms matching the mixed SFT dataset."""

import numpy as np


_IMAGE_KEYS = ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")


def canonical_observation(observation, mirror):
    """Legacy/offline helper. Live rollout conversion belongs to the WS boundary."""
    from expo_ft.env.model_frame import ModelFrame, model_inputs
    frame = ModelFrame({'mirror_images': {'side': mirror, 'wrist': mirror},
                        'mirror_robot_coordinates': mirror})
    return model_inputs(frame.observation(observation))


def physical_action(action, mirror):
    from expo_ft.env.model_frame import ModelFrame
    result = ModelFrame({'mirror_robot_coordinates': mirror}).action(action)
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError('Expected a finite 7D Cartesian action')
    return result


def validate_camera_views(task_config, expected_views):
    """Check the selected eyes before constructing a physical robot environment."""
    for key, expected in expected_views.items():
        if task_config.get(key) != expected:
            raise ValueError(f"Mirror training requires {key}={expected!r}; got {task_config.get(key)!r}")


def validate_eval_task(task_config, expected):
    """Validate the coordinator/client control contract before opening hardware."""
    for key, value in expected.items():
        if task_config.get(key) != value:
            raise ValueError(f"Eval task mismatch for {key}: expected {value!r}, got {task_config.get(key)!r}")
