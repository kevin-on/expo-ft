import os
import select
import sys
import cv2
import numpy as np

# Ensure project root is on path when run as script.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(_SCRIPT_DIR))
)  # real_utils -> client -> project root
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# Color definitions from RGB (stored as BGR for OpenCV).
def _rgb_to_bgr(r, g, b):
    return (b, g, r)


COLOR_BGR = {
    "red": [_rgb_to_bgr(220, 50, 50)],  # fallback for unknown color keys
    "light_plug": [
        _rgb_to_bgr(255, 255, 255),
        _rgb_to_bgr(255, 255, 253),
    ],
}

# Per-color cap on HSV tolerance (h, s, v) for stricter masks where needed.
COLOR_HSV_TOL_CAPS = {}


def _bgr_to_hsv_range(bgr_center, h_tol=30, s_tol=60, v_tol=60):
    """Return (lower, upper) HSV arrays for OpenCV (H 0-180, S/V 0-255)."""
    bgr = np.uint8([[[bgr_center[0], bgr_center[1], bgr_center[2]]]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = int(hsv[0, 0, 0]), int(hsv[0, 0, 1]), int(hsv[0, 0, 2])
    lower = np.array([max(0, h - h_tol), max(0, s - s_tol), max(0, v - v_tol)], dtype=np.uint8)
    upper = np.array([min(180, h + h_tol), min(255, s + s_tol), min(255, v + v_tol)], dtype=np.uint8)
    return lower, upper


def _raw_mask_hsv(image_hsv, color, h_tol=30, s_tol=60, v_tol=60):
    """Build raw inRange mask for color using COLOR_BGR (one or more BGR values = one or more HSV ranges).
    h_tol, s_tol, v_tol: HSV range tolerance around each BGR center (default 30, 60, 60)."""
    key = color.lower()
    if key not in COLOR_BGR:
        key = "red"
    if key in COLOR_HSV_TOL_CAPS:
        cap_h, cap_s, cap_v = COLOR_HSV_TOL_CAPS[key]
        h_tol = min(int(h_tol), cap_h)
        s_tol = min(int(s_tol), cap_s)
        v_tol = min(int(v_tol), cap_v)
    mask = np.zeros(image_hsv.shape[:2], dtype=np.uint8)
    for bgr in COLOR_BGR[key]:
        lower, upper = _bgr_to_hsv_range(bgr, h_tol=h_tol, s_tol=s_tol, v_tol=v_tol)
        mask = cv2.bitwise_or(mask, cv2.inRange(image_hsv, lower, upper))
    return mask


class PickBlocksDetector:
    """Pick-only detector: success from robot state (no camera/depth).

    Success when the gripper stalls inside `success_gripper_range` -- commanded shut but
    held part-way open, i.e. an object is between the fingers -- while it is still closing
    (gripper_velocity > 0.5) and cartesian z is above `success_height`, for
    `consecutive_steps` steps in a row.

    `consecutive_steps` is a step count, not a duration: at `pick`'s 10 Hz the default 6 is
    600 ms, at `dynamic_pick`'s 30 Hz it is 200 ms. Set it per task alongside `control_hz`.
    """

    def __init__(self, success_height=0.2, success_gripper_range=(0.3, 0.5), consecutive_steps=6):
        self._consecutive_condition_steps = 0  # steps where all pick success conditions hold
        self.success_height = float(success_height)
        self.success_gripper_range = tuple(float(v) for v in success_gripper_range)
        self.consecutive_steps = int(consecutive_steps)

    def reset_sequence(self):
        """Reset consecutive pick success counter."""
        self._consecutive_condition_steps = 0

    def detect(
        self,
        gripper_position=None,
        cartesian_position=None,
        gripper_velocity=None,
        debug=False,
        step=0,
    ):
        """True once pick success conditions hold for `consecutive_steps` steps in a row."""
        if gripper_position is not None and cartesian_position is not None:
            gp = float(np.asarray(gripper_position).reshape(-1)[0])
            cp = np.asarray(cartesian_position).reshape(-1)
            z = float(cp[2]) if len(cp) > 2 else 0.0
            gripper_in_range = self.success_gripper_range[0] <= gp <= self.success_gripper_range[1]
            z_high = z > self.success_height
            gripper_closing = gripper_velocity is None or float(gripper_velocity) > 0.5

            if gripper_in_range and z_high and gripper_closing:
                self._consecutive_condition_steps += 1
            else:
                self._consecutive_condition_steps = 0

            if debug:
                print(
                    f"[pick_detector] step={step} gripper={gp:.3f} in_range={gripper_in_range} "
                    f"(range={self.success_gripper_range}) z={z:.3f} z_high={z_high} "
                    f"(height={self.success_height}) closing={gripper_closing} "
                    f"consecutive={self._consecutive_condition_steps}/{self.consecutive_steps}"
                )

        if self._consecutive_condition_steps >= self.consecutive_steps:
            return True

        return False


class LightPlugDetector:
    """Light plug success detector.

    Detects light_plug color pixels in the full frame via HSV color segmentation.
    Success when pixel count drops below ``max_yellow_pixels`` for
    ``consecutive_frames`` in a row (plug has been removed / light is on).
    """

    def __init__(
        self,
        detector_image_key,
        max_yellow_pixels=100,
        consecutive_frames=5,
        detect_size=384,
        h_tol=5,
        s_tol=15,
        v_tol=15,
        **kwargs,
    ):
        self.detector_image_key = detector_image_key
        self.max_yellow_pixels = int(max_yellow_pixels)
        self.consecutive_frames = int(consecutive_frames)
        self.detect_size = int(detect_size) if detect_size is not None else None
        self.h_tol = int(h_tol)
        self.s_tol = int(s_tol)
        self.v_tol = int(v_tol)
        self._consecutive_count = 0

    def _resize_keep_ratio(self, frame):
        if self.detect_size is None:
            return frame
        h, w = frame.shape[:2]
        scale = self.detect_size / max(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _count_plug_pixels(self, frame_hsv):
        mask = _raw_mask_hsv(frame_hsv, "light_plug", h_tol=self.h_tol, s_tol=self.s_tol, v_tol=self.v_tol)
        return int(np.count_nonzero(mask > 0))

    def reset_sequence(self):
        self._consecutive_count = 0

    def detect(self, raw_obs, debug=False, step=0, visualize_path=None):
        """Returns (success, terminate). Terminate is always False.

        Success = light_plug pixel count >= min_plug_pixels for
        consecutive_frames consecutive frames (plug is visible / plugged in).
        """
        images = raw_obs.get("image", {})
        if self.detector_image_key not in images:
            return False, False

        frame = np.asarray(images[self.detector_image_key], dtype=np.uint8)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if self.detect_size is not None:
            frame = self._resize_keep_ratio(frame)

        if frame.size == 0:
            return False, False
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        yellow_count = self._count_plug_pixels(frame_hsv)
        condition_met = yellow_count >= self.max_yellow_pixels

        if condition_met:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        success = self._consecutive_count >= self.consecutive_frames

        if debug:
            print(
                f"[light_plug_detector] step={step} plug_px={yellow_count} "
                f"(min={self.max_yellow_pixels}) condition={condition_met} "
                f"consecutive={self._consecutive_count}/{self.consecutive_frames} "
                f"success={success}"
            )

        return success, False


def success_detector_manual():
    """Non-blocking manual episode override via stdin (when input is pending).

    Returns one of:
    - "keep_going": no input queued, or user chose 3
    - "success": user entered 1
    - "reset": user entered 2 (end episode without success)

    Uses /dev/tty when available so it works even if stdin is redirected.
    """

    def _readline(prompt: str) -> str:
        try:
            sys.stdout.write(prompt + "\n")
            sys.stdout.flush()
        except Exception:
            pass
        # Prefer controlling terminal if available
        try:
            with open("/dev/tty", "r") as tty:
                return tty.readline()
        except Exception:
            # Fallback to stdin
            try:
                return sys.stdin.readline()
            except Exception:
                return False

    # Non-blocking first stage: only proceed to blocking prompt if a line is pending
    def _pending_line() -> bool:
        # Check controlling terminal only; avoid stdin which might always appear ready
        try:
            with open("/dev/tty", "r") as tty:
                fd = tty.fileno()
                rlist, _, _ = select.select([fd], [], [], 0.0)
                return bool(rlist)
        except Exception:
            return False

    if not _pending_line():
        # No user input queued; do not block
        return "keep_going"

    while True:
        choice = _readline("[manual] Enter 1 for success, 2 for reset, 3 for keep going, then ENTER:")
        if not choice:
            return "keep_going"
        choice = choice.strip()
        if choice == "1":
            print("[manual] Success annotated.")
            return "success"
        if choice == "2":
            print("[manual] Doing reset.")
            return "reset"
        if choice == "3":
            print("[manual] Keep going.")
            return "keep_going"
        print("[manual] Invalid input. Please type 1 (success), 2 (reset), or 3 (keep going), then ENTER:")
