"""Camera/robot-free orientation helpers shared by teleop and demo collection."""
import numpy as np
from scipy.spatial.transform import Rotation


# DROID RobotIKSolver scales normalized rotation commands by this amount.
DROID_MAX_ROT_DELTA = .15
# Requested correction per control tick, not a guaranteed physical angular speed.
VERTICAL_ROT_STEP = .015


def vertical_orientation(euler):
    """Tool +Z points down in the base frame; preserve heading about base Z."""
    return np.array([np.pi, 0., euler[2]], dtype=np.float64)


def rotation_step(current, target):
    """Approach the fixed orientation via a small rotation, handling Euler wrap."""
    rotation = Rotation.from_euler('xyz', current)
    error = (Rotation.from_euler('xyz', target) * rotation.inv()).as_rotvec()
    angle = np.linalg.norm(error)
    if angle > VERTICAL_ROT_STEP:
        error *= VERTICAL_ROT_STEP / angle
    return (Rotation.from_rotvec(error) * rotation).as_euler('xyz')


def vertical_velocity_action(action, current, target):
    """Replace mouse rotation with feedback, preserving XYZ and gripper commands.

    Match DROID pose_diff / cartesian_delta_to_velocity so collection keeps its
    existing normalized Cartesian-velocity action format and recording path.
    """
    corrected = np.array(action, dtype=np.float64, copy=True)
    if corrected.shape != (7,) or not np.isfinite(corrected).all():
        raise ValueError('Expected seven finite Cartesian/gripper velocity values')
    next_rotation = Rotation.from_euler('xyz', rotation_step(current, target))
    delta = (next_rotation * Rotation.from_euler('xyz', current).inv()).as_euler('xyz')
    corrected[3:6] = delta / DROID_MAX_ROT_DELTA
    return corrected
