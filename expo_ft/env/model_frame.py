"""Physical/model frame conversion, executed only at the workstation RPC boundary.

Data collection stays physical; offline converters select their own convention.
"""
import numpy as np

IMAGE_KEYS = ('exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left')


def model_inputs(observation):
    """Select policy/replay fields without any coordinate transformation."""
    result = {k: observation[k] for k in (*IMAGE_KEYS, 'cartesian_position', 'gripper_position')}
    if 'prompt' in observation:
        result['prompt'] = observation['prompt']
    return result


class ModelFrame:
    def __init__(self, config=None, *, require_explicit=False):
        config = config or {}
        if set(config) - {'mirror_images', 'mirror_robot_coordinates'}:
            raise ValueError('Unknown model_frame setting')
        images = config.get('mirror_images', {})
        if set(images) - {'side', 'wrist'}:
            raise ValueError('mirror_images accepts side and wrist')
        if require_explicit and (
            'mirror_robot_coordinates' not in config or set(images) != {'side', 'wrist'}
        ):
            raise ValueError(
                'Set model_frame.mirror_images.side, model_frame.mirror_images.wrist, '
                'and model_frame.mirror_robot_coordinates explicitly in the robot config '
                '(use false when no reflection is needed)'
            )
        self.side = images.get('side', False)
        self.wrist = images.get('wrist', False)
        self.coordinates = config.get('mirror_robot_coordinates', False)
        if any(type(x) is not bool for x in (self.side, self.wrist, self.coordinates)):
            raise ValueError('Mirror settings must be booleans')

    def observation(self, observation):
        result = dict(observation)
        for key, flip in zip(IMAGE_KEYS, (self.side, self.side, self.wrist)):
            if flip and key in result:
                result[key] = np.ascontiguousarray(np.asarray(result[key])[:, ::-1, :])
        if self.coordinates:
            pose = np.array(result['cartesian_position'], copy=True)
            if pose.shape != (6,):
                raise ValueError('Model frame requires six-component Cartesian pose')
            pose[[1, 3, 5]] *= -1
            result['cartesian_position'] = pose
        return result

    def action(self, action):
        """Reflection is its own inverse: model→physical or executed physical→model."""
        result = np.array(action, dtype=np.float64, copy=True)
        if self.coordinates:
            if result.shape != (7,):
                raise ValueError('Model frame requires seven-component Cartesian action')
            result[[1, 3, 5]] *= -1
        return result
