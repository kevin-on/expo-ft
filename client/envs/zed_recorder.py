import logging
import os
import threading
import time

import cv2
import imageio
import numpy as np

logger = logging.getLogger(__name__)

# ZED resolution modes by (width, height); each camera model supports a subset.
_ZED_RESOLUTIONS = {
    "HD4K": (3840, 2160),
    "QHDPLUS": (3200, 1800),
    "HD2K": (2208, 1242),
    "HD1536": (1920, 1536),
    "HD1200": (1920, 1200),
    "HD1080": (1920, 1080),
    "HD720": (1280, 720),
    "SVGA": (960, 600),
    "VGA": (672, 376),
}


def _as_bgr(frame):
    """ZED retrieves BGRA; drop alpha and copy (the Mat is reused)."""
    frame = np.asanyarray(frame)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame.copy()


def normalize_zed_serial(serial):
    """"29838012_left" / "29838012" -> "29838012" (the SDK wants the bare serial)."""
    s = str(serial or "").strip()
    view = s.rpartition("_")
    if view[1] and view[2] in ("left", "right"):
        return view[0]
    return s


def release_zed_from_reader(camera_reader, serial):
    """Close `serial` in the DROID MultiCameraWrapper so another opener can take it
    (the SDK allows one opener per serial)."""
    if camera_reader is None:
        return False
    serial = normalize_zed_serial(serial)
    cam = getattr(camera_reader, "camera_dict", {}).pop(serial, None)
    if cam is None:
        return False
    stop_bg = getattr(camera_reader, "stop_background_reading", None)
    if stop_bg is not None:
        stop_bg()
    cam.disable_camera()
    time.sleep(0.5)  # the SDK frees the USB device asynchronously
    logger.info("ZedRecorder: released ZED %s from the policy camera reader", serial)
    return True


def release_unused_zeds(camera_reader, keep_ids):
    """Close every ZED the reader holds that `keep_ids` does not name. The reader opens
    every connected ZED, and the SDK locks each one to this process."""
    keep = {normalize_zed_serial(cam_id) for cam_id in keep_ids if cam_id}
    released = []
    for serial in list(getattr(camera_reader, "camera_dict", {})):
        if serial not in keep and release_zed_from_reader(camera_reader, serial):
            released.append(serial)
    if released:
        logger.info("Released unused ZEDs %s (kept %s)", released, sorted(keep))
    return released


class ZedRecorder:
    """Per-episode MP4 recorder for a dedicated ZED, opened by this class alone.

    A daemon thread owns the camera for the recorder's lifetime and grabs at the native
    rate; start()/stop() only gate whether frames are written, one MP4 per episode.
    """

    def __init__(self, video_dir, serial=None, resolution=None, fps=None, start_timeout=20.0):
        import pyzed.sl as sl

        self.video_dir = video_dir
        self.start_timeout = start_timeout

        devices = list(sl.Camera.get_device_list())
        if not devices:
            raise RuntimeError("record_camera set but no ZED device found")
        serials = [str(d.serial_number) for d in devices]
        serial = normalize_zed_serial(serial)
        if serial in ("", "None", "True", "auto"):
            raise ValueError(
                f"record_camera needs an explicit ZED serial (e.g. '29838012'); connected: {serials}"
            )
        if serial not in serials:
            raise RuntimeError(f"ZED {serial} not found among {serials}")
        self.serial = serial

        # Requested mode; the camera reports what it actually opened at.
        self.requested_resolution = self._resolve_resolution(sl, resolution)
        self.requested_fps = int(fps) if fps else 0  # 0 = SDK default
        self.width = self.height = None
        self.fps = None

        self._thread = None
        self._shutdown = threading.Event()
        self._loop_ready = threading.Event()   # camera opened + warmed up
        self._loop_error = None

        # Per-episode recording window.
        self._record = threading.Event()
        self._first_frame = threading.Event()
        self._writer_closed = threading.Event()
        self._mp4_path = None
        self._ep_idx = 0
        self._frames = 0

        # Open at init so a USB/device failure surfaces before the robot moves.
        self._ensure_thread()

    @staticmethod
    def _resolve_resolution(sl, resolution):
        """An enum name ("HD1080"), a (width, height) pair, or None (= HD1080)."""
        if resolution is None:
            name = "HD1080"
        elif isinstance(resolution, str):
            name = resolution.upper()
        else:
            wh = tuple(int(v) for v in resolution)
            match = [k for k, v in _ZED_RESOLUTIONS.items() if v == wh]
            if not match:
                raise ValueError(
                    f"ZED record_resolution {wh} is not a ZED mode; known modes: "
                    f"{sorted(_ZED_RESOLUTIONS.items(), key=lambda kv: -kv[1][0])}"
                )
            name = match[0]
        if not hasattr(sl.RESOLUTION, name):
            raise ValueError(f"ZED resolution {name!r} is unknown to this SDK build")
        return getattr(sl.RESOLUTION, name)

    def _ensure_thread(self):
        """Start the camera thread once and wait until the camera is streaming."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown.clear()
        self._loop_ready.clear()
        self._loop_error = None
        self._thread = threading.Thread(target=self._camera_loop, name="zed-recorder", daemon=True)
        self._thread.start()
        if not self._loop_ready.wait(timeout=self.start_timeout):
            self._shutdown.set()
            err = self._loop_error
            raise RuntimeError(
                f"ZedRecorder: camera {self.serial} did not start streaming within "
                f"{self.start_timeout:.0f}s" + (f": {err}" if err else "")
            )
        if self._loop_error is not None:
            raise RuntimeError(
                f"ZedRecorder: camera {self.serial} failed to open: {self._loop_error}"
                + self._bandwidth_hint()
            )

    def _bandwidth_hint(self):
        if "LOW USB BANDWIDTH" not in str(self._loop_error):
            return ""
        return (
            "\n  The ZEDs share one USB controller and the requested record mode does not "
            "fit alongside the policy cameras. Either lower record_resolution/record_fps "
            "in the task config, or move this camera to a USB3 port on a different "
            "controller (in `lsusb -t`, each root hub line is one controller)."
        )

    def start(self):
        if self._record.is_set():
            return
        os.makedirs(self.video_dir, exist_ok=True)
        self._ensure_thread()
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._mp4_path = os.path.join(self.video_dir, f"record_{ts}_ep{self._ep_idx:06d}.mp4")
        self._frames = 0
        self._first_frame.clear()
        self._writer_closed.clear()
        self._record.set()  # mp4_path is set before the flag the camera thread gates on
        if not self._first_frame.wait(timeout=self.start_timeout):
            self._record.clear()
            err = self._loop_error
            raise RuntimeError(
                f"ZedRecorder: no frame from {self.serial} within "
                f"{self.start_timeout:.0f}s of start()" + (f": {err}" if err else "")
            )
        logger.info("ZedRecorder: recording -> %s", self._mp4_path)
        self._ep_idx += 1

    def _camera_loop(self):
        import pyzed.sl as sl

        cam = sl.Camera()
        writer = None
        opened = False
        try:
            init = sl.InitParameters(
                camera_resolution=self.requested_resolution,
                camera_fps=self.requested_fps,
                depth_mode=sl.DEPTH_MODE.NONE,  # video only; depth is the expensive part of grab()
                depth_stabilization=False,
                camera_image_flip=sl.FLIP_MODE.OFF,
            )
            init.set_from_serial_number(int(self.serial))
            status = cam.open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"ZED {self.serial} failed to open: {status}")
            opened = True

            conf = cam.get_camera_information().camera_configuration
            self.width = int(conf.resolution.width)
            self.height = int(conf.resolution.height)
            self.fps = int(round(conf.fps)) or 30
            logger.info(
                "ZedRecorder: %s %dx%d@%dfps", self.serial, self.width, self.height, self.fps
            )

            runtime = sl.RuntimeParameters()
            image = sl.Mat()
            for _ in range(15):  # let auto-exposure settle
                if self._shutdown.is_set():
                    return
                cam.grab(runtime)
            self._loop_ready.set()

            while not self._shutdown.is_set():
                recording = self._record.is_set()
                if recording and writer is None:
                    # macro_block_size=2 keeps the native frame size.
                    writer = imageio.get_writer(
                        self._mp4_path, fps=self.fps, codec="libx264", macro_block_size=2,
                        pixelformat="yuv420p",
                        ffmpeg_params=["-preset", "veryfast", "-crf", "20"],
                    )
                elif not recording and writer is not None:
                    writer.close()
                    writer = None
                    self._writer_closed.set()

                if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    continue
                if writer is None or not self._record.is_set():
                    continue
                cam.retrieve_image(image, sl.VIEW.LEFT)
                writer.append_data(cv2.cvtColor(_as_bgr(image.get_data()), cv2.COLOR_BGR2RGB))
                self._frames += 1
                self._first_frame.set()
        except Exception as e:
            self._loop_error = e
            logger.error("ZedRecorder: camera loop error: %s", e)
        finally:
            if writer is not None:
                writer.close()
            self._writer_closed.set()
            self._loop_ready.set()
            if opened:
                try:
                    cam.close()
                except Exception:
                    pass

    def stop(self):
        if not self._record.is_set():
            return
        self._record.clear()
        self._writer_closed.wait(timeout=10.0)  # the camera thread closes the writer
        if self._loop_error is not None:
            raise RuntimeError(f"ZedRecorder: camera loop failed: {self._loop_error}")
        if self._frames == 0 or not os.path.exists(self._mp4_path) or os.path.getsize(self._mp4_path) == 0:
            raise RuntimeError(
                f"ZedRecorder: no video saved at {self._mp4_path} ({self._frames} frames)"
            )
        logger.info("ZedRecorder: wrote %s (%d frames)", self._mp4_path, self._frames)

    def close(self):
        try:
            self.stop()
        except Exception:
            pass
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
