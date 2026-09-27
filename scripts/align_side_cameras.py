#!/usr/bin/env python3
"""Live mirror/opacity alignment for side and wrist ZED pairs (no robot connection).

Run on the workstation desktop with client/.venv/bin/python. Separate side and
wrist windows each show camera 1 LEFT + camera 2 RIGHT and camera 1 RIGHT +
camera 2 LEFT overlays, with all four unflipped source images below.
Camera 1 belongs to robot0; camera 2 belongs to robot1. Use --side-only for
the original two-camera view.
"""

import argparse
from dataclasses import dataclass, replace
import os
import sys
import threading
import time

import cv2
import numpy as np


GUIDES = "Guides: 0=off 1=cross 2=grid"
FLIP_NAMES = ("none", "side1", "side2")
STALE_AFTER = 0.75


@dataclass(frozen=True)
class CameraView:
    left: np.ndarray | None = None
    right: np.ndarray | None = None
    updated_at: float = 0.0
    issue: str = "Opening camera..."
    fatal: bool = False

    def is_live(self, now):
        return (self.left is not None and self.right is not None
                and not self.issue and now - self.updated_at < STALE_AFTER)


class ZedReader:
    """One camera-owning worker; publishes complete stereo pairs from one grab."""

    def __init__(self, sdk, serial, fps):
        self.sdk = sdk
        self.serial = serial
        self.fps = fps
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self._view = CameraView()
        self.thread = threading.Thread(target=self._run, name=f"zed-{serial}", daemon=True)

    def snapshot(self):
        with self._lock:
            return self._view

    def _set_issue(self, message, fatal=False):
        with self._lock:
            self._view = replace(self._view, issue=message, fatal=fatal)

    def _run(self):
        sl = self.sdk
        camera = None
        images = []
        try:
            params = sl.InitParameters()
            params.set_from_serial_number(self.serial)
            params.camera_resolution = sl.RESOLUTION.HD1080
            params.camera_fps = self.fps
            params.depth_mode = sl.DEPTH_MODE.NONE
            params.camera_image_flip = sl.FLIP_MODE.OFF
            params.open_timeout_sec = 5.0
            # Return grab errors promptly while SDK recovery runs in the background.
            params.async_grab_camera_recovery = True
            camera = sl.Camera()
            result = camera.open(params)
            if result != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(
                    f"open failed: {result}. Check USB/permissions and close any viewer using this camera."
                )
            images = [sl.Mat(), sl.Mat()]
            runtime = sl.RuntimeParameters()
            opened_at = time.monotonic()
            self._set_issue("Waiting for first stereo pair...")
            while not self.stop.is_set():
                result = camera.grab(runtime)
                if result == sl.ERROR_CODE.SUCCESS:
                    # Both lenses must come from this same successful grab.
                    for eye, image in zip((sl.VIEW.LEFT, sl.VIEW.RIGHT), images):
                        result = camera.retrieve_image(image, eye, sl.MEM.CPU)
                        if result != sl.ERROR_CODE.SUCCESS:
                            break
                if result != sl.ERROR_CODE.SUCCESS:
                    self._set_issue(str(result))
                    if self.snapshot().left is None and time.monotonic() - opened_at > 10:
                        raise RuntimeError(f"no first stereo pair within 10 seconds: {result}")
                    self.stop.wait(0.03)
                    continue
                # Own both BGR arrays before the SDK reuses its Mats. Never publish
                # a new left frame paired with an old or failed right frame.
                left, right = [cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR) for image in images]
                with self._lock:
                    self._view = CameraView(left=left, right=right, updated_at=time.monotonic(), issue="")
        except Exception as exc:
            self._set_issue(str(exc), fatal=True)
        finally:
            try:
                for image in images:
                    image.free(sl.MEM.CPU)
            finally:
                if camera is not None:
                    camera.close()


def blend_frames(side1, side2, flip, opacity):
    """Mirror at most one input, then blend. Inputs remain unchanged."""
    if flip == 1:
        side1 = cv2.flip(side1, 1)
    elif flip == 2:
        side2 = cv2.flip(side2, 1)
    return cv2.addWeighted(side1, 1.0 - opacity, side2, opacity, 0.0)


def draw_guides(frame, mode):
    if not mode:
        return
    height, width = frame.shape[:2]
    fractions = (0.5,) if mode == 1 else (0.25, 0.5, 0.75)
    for fraction in fractions:
        x, y = round((width - 1) * fraction), round((height - 1) * fraction)
        for start, end in [((x, 0), (x, height - 1)), ((0, y), (width - 1, y))]:
            cv2.line(frame, start, end, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.line(frame, start, end, (150, 220, 220), 1, cv2.LINE_AA)


def caption(text, width, color=(225, 225, 225)):
    strip = np.full((30, width, 3), 28, dtype=np.uint8)
    cv2.putText(strip, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return strip


def render(views, serials, width, flip, opacity, guides, now, camera_label="Side"):
    panel_width = width // 2
    height = round(panel_width * 9 / 16)
    live = [view.is_live(now) for view in views]
    # Each column's raw previews sit directly beneath their corresponding overlay.
    pairs = (
        (views[0].left, views[1].right, f"{camera_label} 1 LEFT", f"{camera_label} 2 RIGHT"),
        (views[0].right, views[1].left, f"{camera_label} 1 RIGHT", f"{camera_label} 2 LEFT"),
    )
    panels = []
    for first, second, first_label, second_label in pairs:
        frames = [
            cv2.resize(frame, (panel_width, height), interpolation=cv2.INTER_AREA)
            if frame is not None else np.zeros((height, panel_width, 3), dtype=np.uint8)
            for frame in (first, second)
        ]
        if all(live):
            overlay = blend_frames(frames[0], frames[1], flip, opacity)
            draw_guides(overlay, guides)
        else:
            overlay = np.zeros((height, panel_width, 3), dtype=np.uint8)
            cv2.putText(overlay, "Waiting for two live cameras",
                        (15, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (80, 180, 255), 1, cv2.LINE_AA)
        previews = []
        for index, (frame, label) in enumerate(zip(frames, (first_label, second_label))):
            preview = cv2.resize(frame, (panel_width // 2, height // 2), interpolation=cv2.INTER_AREA)
            # Stale source previews are dimmed and labeled; neither overlay uses them.
            if not live[index]:
                preview = (preview * 0.35).astype(np.uint8)
                label += " [STALE / WAITING]"
            color = (140, 230, 140) if live[index] else (80, 180, 255)
            previews.append(np.vstack([caption("RAW: " + label, panel_width // 2, color), preview]))
        panels.append(np.vstack([
            caption(first_label + " + " + second_label, panel_width),
            overlay,
            np.hstack(previews),
        ]))
    statuses = []
    for index, (view, serial) in enumerate(zip(views, serials)):
        if view.left is None or view.right is None:
            status = "WAITING"
        elif not live[index]:
            status = f"STALE ({now - view.updated_at:.1f}s)"
        else:
            status = f"{(now - view.updated_at) * 1000:.0f} ms old"
        color = (140, 230, 140) if live[index] else (80, 180, 255)
        statuses.append(caption(f"{camera_label} {index + 1} | {serial} | {status}", panel_width, color))
    mirror = "none" if flip == 0 else f"{camera_label} {flip}"
    title = f"BOTH OVERLAYS | Mirror: {mirror} | {camera_label} 1: {1-opacity:.0%} / {camera_label} 2: {opacity:.0%}"
    issues = "; ".join(f"{camera_label} {i+1}: {v.issue}" for i, v in enumerate(views) if v.issue)
    footer = issues or "Sliders: this window | F/G: mirror/guides in all windows | Q / Esc: quit"
    return np.vstack([caption(title, width), np.hstack(panels), np.hstack(statuses), caption(footer, width)])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side1", type=int, default=38651013, help="Side 1 ZED serial (default: 38651013)")
    parser.add_argument("--side2", type=int, default=29838012, help="Side 2 ZED serial (default: 29838012)")
    parser.add_argument("--wrist1", type=int, default=15577469, help="Robot0 wrist ZED serial (default: 15577469)")
    parser.add_argument("--wrist2", type=int, default=12841040, help="Robot1 wrist ZED serial (default: 12841040)")
    parser.add_argument("--side-only", action="store_true", help="Open only the original side camera pair")
    parser.add_argument("--flip", choices=FLIP_NAMES, default="side2", help="Initial mirror selection: none, robot0 (side1), robot1 (side2); applies to both windows")
    parser.add_argument("--opacity", type=float, default=0.5, help="Initial camera 2 contribution in each window, 0..1 (default: 0.5)")
    parser.add_argument("--fps", type=int, choices=(15, 30), default=15, help="HD1080 capture FPS per camera (default: 15)")
    parser.add_argument("--width", type=int, default=1600, help="Total display width >=960, multiple of 4; capture stays at HD1080 (default: 1600)")
    args = parser.parse_args(argv)
    serials = [args.side1, args.side2] + ([] if args.side_only else [args.wrist1, args.wrist2])
    if any(serial <= 0 for serial in serials) or len(set(serials)) != len(serials):
        parser.error("All enabled camera serial numbers must be positive and distinct")
    if not 0 <= args.opacity <= 1:
        parser.error("--opacity must be between 0 and 1")
    if args.width < 960 or args.width % 4:
        parser.error("--width must be a multiple of 4 and >=960")
    return args


def window_controls(label):
    return (f"{label} camera alignment", f"{label} 2 opacity %",
            f"Mirror: 0=off 1={label.lower()}1 2={label.lower()}2")


def run(args, sdk):
    pairs = [("Side", (args.side1, args.side2))]
    if not args.side_only:
        pairs.append(("Wrist", (args.wrist1, args.wrist2)))
    readers = {serial: ZedReader(sdk, serial, args.fps) for _, serials in pairs for serial in serials}
    started = []
    try:
        # Create the GUI before opening devices so a window failure cannot leave readers running.
        for label, serials in pairs:
            window, opacity_control, flip_control = window_controls(label)
            cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(window, args.width, round(args.width * 27 / 64) + 210)
            cv2.createTrackbar(opacity_control, window, round(args.opacity * 100), 100, lambda _: None)
            cv2.createTrackbar(flip_control, window, FLIP_NAMES.index(args.flip), 2, lambda _: None)
            cv2.createTrackbar(GUIDES, window, 2, 2, lambda _: None)
            print(f"{label} 1 (robot0): {serials[0]}; {label} 2 (robot1): {serials[1]}")
        for reader in readers.values():
            reader.thread.start()
            started.append(reader)
        print(f"HD1080 @ {args.fps} FPS, LEFT + RIGHT views, depth off.")
        print("Each window: camera 1 LEFT + camera 2 RIGHT; camera 1 RIGHT + camera 2 LEFT.")
        print("Sliders affect their own window. F/G cycle mirror/guides in all windows.")
        print("Q/Esc, closing either window, or Ctrl-C exits and releases all cameras.")
        while True:
            for label, serials in pairs:
                window, opacity_control, flip_control = window_controls(label)
                views = [readers[serial].snapshot() for serial in serials]
                for serial, view in zip(serials, views):
                    if view.fatal:
                        raise RuntimeError(f"{label} ZED {serial}: {view.issue}")
                opacity = cv2.getTrackbarPos(opacity_control, window) / 100.0
                flip = cv2.getTrackbarPos(flip_control, window)
                guides = cv2.getTrackbarPos(GUIDES, window)
                cv2.imshow(window, render(views, serials, args.width,
                                         flip, opacity, guides, time.monotonic(), label))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            try:
                if any(cv2.getWindowProperty(window_controls(label)[0], cv2.WND_PROP_VISIBLE) < 1
                       for label, _ in pairs):
                    break
            except cv2.error:
                break  # The native window was closed.
            for label, _ in pairs:
                window, _, flip_control = window_controls(label)
                if key in (ord("f"), ord("F")):
                    cv2.setTrackbarPos(flip_control, window, (cv2.getTrackbarPos(flip_control, window) + 1) % 3)
                elif key in (ord("g"), ord("G")):
                    cv2.setTrackbarPos(GUIDES, window, (cv2.getTrackbarPos(GUIDES, window) + 1) % 3)
    finally:
        for reader in started:
            reader.stop.set()
        for reader in started:
            reader.thread.join(timeout=6)
            if reader.thread.is_alive():
                print(f"Warning: ZED {reader.serial} is still shutting down; SDK worker did not return.", file=sys.stderr)
        cv2.destroyAllWindows()


def main(argv=None):
    args = parse_args(argv)  # --help and argument errors never import the SDK or open devices.
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        print("Error: DISPLAY is not set. Run in a terminal on the workstation's graphical desktop.", file=sys.stderr)
        return 1
    try:
        import pyzed.sl as sl

        run(args, sl)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
