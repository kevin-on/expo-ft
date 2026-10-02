#!/usr/bin/env python3
"""Align side and wrist ZED pairs in separate desktop windows (no robot connection).

Each window shows robot0, robot1, and their overlay. Choose one LEFT/RIGHT lens
per robot, mirror either or both independently, and adjust robot1's blend weight.
With --dataset-root, a second row compares each robot with recorded stereo
frames. All three overlays have independent opacity controls. Choose dataset
episodes and frames in the UI. No implicit dataset path is used.
Defaults come from configs/robots/robot-{0,1}.json; controls never edit those files.
Run from the workstation desktop with client/.venv/bin/python.
"""

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np


STALE_AFTER = 0.75
REPO = Path(__file__).resolve().parents[1]


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



@dataclass(frozen=True)
class CameraPair:
    name: str
    serials: tuple[int, int]
    eyes: tuple[str, str]


def preview_frames(views, eyes, mirrors, size, opacity, now):
    """Return selected/transformed BGR previews and overlay; never mutate captures.

    A stale source can be inspected dimmed, but cannot contribute to an overlay.
    """
    live = [view.is_live(now) for view in views]
    frames = []
    for view, eye, mirror, healthy in zip(views, eyes, mirrors, live):
        source = getattr(view, eye)
        frame = (cv2.resize(source, size, interpolation=cv2.INTER_AREA) if source is not None
                 else np.zeros((size[1], size[0], 3), dtype=np.uint8))
        if mirror:
            frame = cv2.flip(frame, 1)
        if not healthy:
            frame = (frame * 0.3).astype(np.uint8)
        frames.append(frame)
    overlay = (cv2.addWeighted(frames[0], 1 - opacity, frames[1], opacity, 0)
               if all(live) else np.zeros_like(frames[0]))
    return [*frames, overlay], live


@dataclass(frozen=True)
class ReferenceFrame:
    path: Path
    index: int
    count: int
    left: np.ndarray
    right: np.ndarray

    def snapshot(self):
        return CameraView(self.left, self.right, time.monotonic(), "")


class DatasetReferences:
    """Read individual physical stereo frames; never load a whole trajectory."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.episodes = []
        for robot in (0, 1):
            paths = sorted((p for p in (self.root / f'robot{robot}' / 'success').glob('*/traj.hdf5')
                            if p.parent.name.isdigit()), key=lambda p: (int(p.parent.name), p.parent.name))
            if not paths:
                raise ValueError(f'No successful trajectories: {self.root}/robot{robot}/success/*/traj.hdf5')
            self.episodes.append({p.parent.name: p for p in paths})

    def read(self, robot, camera, episode, index):
        import h5py

        path = self.episodes[robot][episode]
        prefix = {'side': 'exterior_image_1', 'wrist': 'wrist_image'}[camera]
        keys = [f'saved_observation/{prefix}_{eye}' for eye in ('left', 'right')]
        with h5py.File(path, 'r') as file:
            if any(key not in file for key in keys):
                raise ValueError(f'{path}: missing physical {camera} LEFT/RIGHT images')
            arrays = [file[key] for key in keys]
            if any(a.dtype != np.uint8 or a.ndim != 4 or a.shape[-1] != 3 for a in arrays):
                raise ValueError(f'{path}: expected uint8 [frames, height, width, RGB]')
            count = len(arrays[0])
            if count != len(arrays[1]) or not 0 <= index < count:
                raise ValueError(f'{path}: invalid frame {index}; matching stereo frames 0..{count-1} required')
            frames = []
            for a in arrays:
                h, w = a.shape[1:3]
                if h == 0 or abs(w / h - 16 / 9) > .01:
                    raise ValueError(f'{path}: need uncropped 16:9 images for HD1080 alignment')
                frames.append(cv2.cvtColor(a[index], cv2.COLOR_RGB2BGR))
        return ReferenceFrame(path, index, count, *frames)


class BlendControl:
    """One independent opacity slider, numeric entry and precise adjustments."""

    def __init__(self, parent, labels, value=.5):
        import tkinter as tk
        from tkinter import ttk

        self.labels = labels
        self.value = tk.DoubleVar(value=round(value * 100))
        self.percent = tk.StringVar()
        self.weights = tk.StringVar()
        ttk.Label(parent, textvariable=self.weights).pack(anchor='w')
        self.scale = ttk.Scale(parent, from_=0, to=100, variable=self.value, command=self.set)
        self.scale.pack(fill='x', pady=(5, 4))
        self.scale.bind('<Button-1>', self.seek)
        self.scale.bind('<B1-Motion>', self.seek)
        self.scale.bind('<Left>', lambda _: self.nudge(-1))
        self.scale.bind('<Right>', lambda _: self.nudge(1))
        entry = ttk.Frame(parent)
        entry.pack(fill='x')
        ttk.Button(entry, text='−', width=2, command=lambda: self.nudge(-1)).pack(side='left')
        spin = ttk.Spinbox(entry, from_=0, to=100, increment=1, width=4,
                           textvariable=self.percent, command=self.typed)
        spin.pack(side='left', padx=3)
        spin.bind('<Return>', self.typed)
        spin.bind('<FocusOut>', self.typed)
        ttk.Label(entry, text='%').pack(side='left')
        ttk.Button(entry, text='+', width=2, command=lambda: self.nudge(1)).pack(side='left', padx=3)
        for percent in (0, 50, 100):
            ttk.Button(entry, text=str(percent), width=3,
                       command=lambda v=percent: self.set(v)).pack(side='left', padx=1)
        self.set(value * 100)

    def set(self, value):
        value = max(0, min(100, round(float(value))))
        self.value.set(value)
        self.percent.set(str(value))
        self.weights.set(f'{self.labels[0]}: {100-value}%   +   {self.labels[1]}: {value}%')

    def typed(self, _event=None):
        try:
            value = float(self.percent.get())
            if not np.isfinite(value):
                raise ValueError('nonfinite weight')
        except ValueError:
            value = self.value.get()
        self.set(value)
        return 'break'

    def nudge(self, amount):
        self.set(self.value.get() + amount)
        return 'break'

    def seek(self, event):
        self.scale.focus_set()
        self.set((event.x - 10) / max(1, self.scale.winfo_width() - 20) * 100)
        return 'break'


class AlignmentWindow:
    """One side/wrist window: two live sources, cross-robot and dataset overlays."""

    def __init__(self, root, pair, readers, args, close, dataset=None):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk

        self.Image, self.ImageTk = Image, ImageTk
        self.pair, self.readers, self.dataset = pair, readers, dataset
        self.window = tk.Toplevel(root)
        self.window.title(f'{pair.name} align')
        desired_height = round(args.width / 3 * 9 / 16) * (2 if dataset else 1) + (320 if dataset else 250)
        self.window.geometry(f'{args.width}x{min(desired_height, root.winfo_screenheight()-70)}')
        self.window.minsize(960, 650 if dataset else 390)
        self.window.protocol('WM_DELETE_WINDOW', close)
        self.window.bind('<Escape>', lambda _: close())
        self.eyes = [tk.StringVar(value=eye) for eye in pair.eyes]
        self.mirrors = [tk.BooleanVar(value=value) for value in args.mirrors]
        self.guide = tk.StringVar(value='Grid')
        self.footer = tk.StringVar()
        self.status = [tk.StringVar() for _ in range(5 if dataset else 3)]
        self.photos, self.items, self.canvases = [], [], []
        self.references = [None, None]
        self.reference_errors = ['', '']
        self.episode_vars, self.frame_vars, self.frame_spins, self.reference_labels = [], [], [], []
        self.blends = []
        self.last_frames = None

        body = ttk.Frame(self.window, padding=12)
        body.pack(fill='both', expand=True)
        header = ttk.Frame(body)
        header.pack(fill='x', pady=(0, 8))
        ttk.Label(header, text=f'{pair.name} alignment', style='Title.TLabel').pack(side='left')
        ttk.Combobox(header, textvariable=self.guide, values=('Off', 'Cross', 'Grid'),
                     state='readonly', width=7).pack(side='right')
        ttk.Label(header, text='Guides').pack(side='right', padx=(16, 5))
        cards = ttk.Frame(body)
        cards.pack(fill='both', expand=True)
        cards.rowconfigure(0, weight=1, uniform='row')
        if dataset:
            cards.rowconfigure(1, weight=1, uniform='row')
        for i in range(3):
            cards.columnconfigure(i, weight=1, uniform='camera')
        titles = ['Robot 0 LIVE', 'Robot 1 LIVE', 'Robot 0 + Robot 1']
        if dataset:
            titles += ['Robot 0: dataset + live', 'Robot 1: dataset + live']
        for i, title in enumerate(titles):
            card = ttk.LabelFrame(cards, text=title, padding=6)
            card.grid(row=i // 3, column=i % 3, sticky='nsew', padx=3, pady=3)
            controls = ttk.Frame(card, height=88)
            controls.pack(fill='x')
            controls.pack_propagate(False)
            if i < 2:
                ttk.Label(controls, text=f'ZED {pair.serials[i]}').pack(anchor='w')
                lenses = ttk.Frame(controls)
                lenses.pack(anchor='w', pady=5)
                for eye in ('left', 'right'):
                    ttk.Radiobutton(lenses, text=eye.upper(), variable=self.eyes[i],
                                    value=eye).pack(side='left', padx=(0, 12))
                ttk.Checkbutton(controls, text='Mirror horizontally', variable=self.mirrors[i]).pack(anchor='w')
            else:
                self.blends.append(BlendControl(controls, ('Robot 0', 'Robot 1') if i == 2 else ('Dataset', 'Live'),
                                                args.opacity if i == 2 else .5))
            canvas = tk.Canvas(card, background='#11161c', highlightthickness=0, height=130)
            canvas.pack(fill='both', expand=True, pady=4)
            self.items.append(canvas.create_image(0, 0, anchor='center'))
            self.canvases.append(canvas)
            self.photos.append(None)
            ttk.Label(card, textvariable=self.status[i], wraplength=max(270, args.width // 3 - 40)).pack(anchor='w')
        if dataset:
            settings = ttk.LabelFrame(cards, text='Dataset reference', padding=10)
            settings.grid(row=1, column=2, sticky='nsew', padx=3, pady=3)
            ttk.Label(settings, text=str(dataset.root), wraplength=args.width // 3 - 55).pack(anchor='w', pady=(0, 5))
            for robot in (0, 1):
                group = ttk.LabelFrame(settings, text=f'Robot {robot}', padding=4)
                group.pack(fill='x', pady=3)
                row = ttk.Frame(group)
                row.pack(fill='x')
                ep = tk.StringVar(value=next(iter(dataset.episodes[robot])))
                frame = tk.StringVar(value='0')
                self.episode_vars.append(ep)
                self.frame_vars.append(frame)
                ttk.Label(row, text='Episode').pack(side='left')
                ttk.Combobox(row, values=tuple(dataset.episodes[robot]), textvariable=ep,
                             state='readonly', width=6).pack(side='left', padx=4)
                ttk.Label(row, text='Frame').pack(side='left')
                spin = ttk.Spinbox(row, from_=0, to=0, textvariable=frame, width=5)
                spin.pack(side='left', padx=4)
                spin.bind('<Return>', lambda _, r=robot: self.load_reference(r))
                self.frame_spins.append(spin)
                ttk.Button(row, text='Load', width=5, command=lambda r=robot: self.load_reference(r)).pack(side='left')
                label = tk.StringVar()
                self.reference_labels.append(label)
                ttk.Label(group, textvariable=label, wraplength=args.width // 3 - 65).pack(anchor='w')
                self.load_reference(robot, initial=True)
            ttk.Label(settings, text='Frame numbers start at 0. Select episode/frame, then Load.\n'
                      'Dataset and live share the lens + mirror controls above.\n'
                      'Wrist comparison needs the same robot pose.',
                      wraplength=args.width // 3 - 55).pack(anchor='w', pady=5)
        ttk.Label(body, textvariable=self.footer).pack(anchor='w', pady=(6, 0))

    def load_reference(self, robot, initial=False):
        try:
            ref = self.dataset.read(robot, self.pair.name.lower(), self.episode_vars[robot].get(),
                                    int(self.frame_vars[robot].get()))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.references[robot] = None  # Never silently compare against a previous selection.
            self.reference_errors[robot] = str(exc)
            self.reference_labels[robot].set(f'Load failed: {exc}')
            if initial:
                raise
        else:
            self.references[robot] = ref
            self.reference_errors[robot] = ''
            self.frame_spins[robot].configure(to=ref.count-1)
            self.reference_labels[robot].set(f'Loaded episode {ref.path.parent.name} / frame {ref.index} of {ref.count}')
        return 'break'

    def refresh(self):
        views = [reader.snapshot() for reader in self.readers]
        now = time.monotonic()
        width = max(1, min(c.winfo_width() for c in self.canvases))
        height = max(1, min(c.winfo_height() for c in self.canvases))
        width = max(1, min(width, round(height * 16 / 9)))
        size = (width, max(1, round(width * 9 / 16)))
        eyes = [v.get() for v in self.eyes]
        mirrors = [v.get() for v in self.mirrors]
        frames, live = preview_frames(views, eyes, mirrors, size, self.blends[0].value.get()/100, now)
        healthy = [*live, all(live)]
        for robot, (view, ok) in enumerate(zip(views, live)):
            label = eyes[robot].upper() + (' · mirrored' if mirrors[robot] else ' · original')
            self.status[robot].set(f'LIVE · {1000*(now-view.updated_at):.0f} ms old · {label}' if ok else
                                   f'{"ERROR" if view.fatal else "WAITING / STALE"} · {view.issue or "No recent frame"}')
        self.status[2].set('LIVE · mirror alignment' if all(live) else 'Paused: needs two live cameras')
        if self.dataset:
            for robot, ref in enumerate(self.references):
                if ref is not None:
                    # Same physical lens and reflection on BOTH inputs: robot1 is not double-flipped.
                    comparison, valid = preview_frames([ref.snapshot(), views[robot]], [eyes[robot]]*2,
                                                       [mirrors[robot]]*2, size,
                                                       self.blends[robot+1].value.get()/100, time.monotonic())
                    frames.append(comparison[2])
                    healthy.append(all(valid))
                    self.status[robot+3].set(f'Episode {ref.path.parent.name} · frame {ref.index} · {eyes[robot].upper()}'
                                              if all(valid) else 'Paused: waiting for live camera')
                else:
                    frames.append(np.zeros_like(frames[0]))
                    healthy.append(False)
                    self.status[robot+3].set('No reference loaded; see Dataset reference error')
        self.last_frames = frames
        mode = ('Off', 'Cross', 'Grid').index(self.guide.get())
        for i, (canvas, frame) in enumerate(zip(self.canvases, frames)):
            if healthy[i]:
                draw_guides(frame, mode)
            photo = self.ImageTk.PhotoImage(self.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), master=self.window)
            self.photos[i] = photo
            canvas.itemconfigure(self.items[i], image=photo)
            canvas.coords(self.items[i], canvas.winfo_width()/2, canvas.winfo_height()/2)
        skew = abs(views[0].updated_at - views[1].updated_at)*1000
        self.footer.set((f'HD1080 · Pair age difference: {skew:.0f} ms · ' if all(live) else '')
                        + 'Preview only; dataset/configs unchanged. Esc or closing either window exits.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-root', type=Path,
                        help='Enable dataset comparisons; root containing robot0/1/success/<episode>/traj.hdf5')
    parser.add_argument('--check-reference', action='store_true',
                        help='With --dataset-root, check initial references without GUI/SDK/devices')
    for robot in (0, 1):
        parser.add_argument(f"--robot{robot}-config", type=Path,
                            default=REPO / f"configs/robots/robot-{robot}.json")
    for name in ("side1", "side2", "wrist1", "wrist2"):
        parser.add_argument(f"--{name}", type=int, help="Override ZED serial (1=robot0, 2=robot1)")
    only = parser.add_mutually_exclusive_group()
    only.add_argument("--side-only", action="store_true")
    only.add_argument("--wrist-only", action="store_true")
    parser.add_argument("--flip", choices=("none", "robot0", "robot1", "both", "side1", "side2"),
                        default="robot1", help="Initial mirror selection; each window's checkboxes are independent")
    parser.add_argument("--opacity", type=float, default=0.5, help="Robot1 blend weight, 0..1")
    parser.add_argument("--fps", type=int, choices=(15, 30), default=15, help="HD1080 capture FPS")
    parser.add_argument("--width", type=int, default=1440, help="Initial width of each window (>=960)")
    args = parser.parse_args(argv)
    if args.check_reference and args.dataset_root is None:
        parser.error('--check-reference requires --dataset-root')
    if not 0 <= args.opacity <= 1:
        parser.error("--opacity must be between 0 and 1")
    if args.width < 960:
        parser.error("--width must be >=960")
    names = ["wrist"] if args.wrist_only else ["side"] if args.side_only else ["side", "wrist"]
    args.pairs = []
    try:
        configs = [json.loads(getattr(args, f"robot{i}_config").read_text()) for i in (0, 1)]
        for name in names:
            serials, eyes = [], []
            for i, config in enumerate(configs):
                serial, eye = config[f"{name}_camera_id"].rsplit("_", 1)
                if eye not in ("left", "right"):
                    raise ValueError(f"Invalid lens in robot{i} {name}_camera_id: {eye}")
                override = getattr(args, f"{name}{i+1}")
                serials.append(int(serial) if override is None else override)
                eyes.append(eye)
            args.pairs.append(CameraPair(name.title(), tuple(serials), tuple(eyes)))
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
        parser.error(f"Invalid robot camera configuration: {exc}")
    serials = [s for pair in args.pairs for s in pair.serials]
    if any(s <= 0 for s in serials) or len(set(serials)) != len(serials):
        parser.error("All enabled camera serial numbers must be positive and distinct")
    args.mirrors = (args.flip in ("robot0", "side1", "both"), args.flip in ("robot1", "side2", "both"))
    return args


def run(args, sdk, dataset=None):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.withdraw()
    readers, started = {}, []
    failure = []
    try:
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font=("sans", 11))
        style.configure("Title.TLabel", font=("sans", 17, "bold"))
        style.configure("TLabelframe.Label", font=("sans", 12, "bold"))
        windows = []
        for pair in args.pairs:
            pair_readers = []
            for serial in pair.serials:
                readers[serial] = ZedReader(sdk, serial, args.fps)
                pair_readers.append(readers[serial])
            windows.append(AlignmentWindow(root, pair, pair_readers, args, root.quit, dataset=dataset))
            print(f"{pair.name}: robot0 {pair.serials[0]}_{pair.eyes[0]}, robot1 {pair.serials[1]}_{pair.eyes[1]}")
        # UI creation (including Pillow/Tk) must succeed before opening any device.
        root.update_idletasks()
        for reader in readers.values():
            reader.thread.start()
            started.append(reader)
        print("Each window: select one lens per robot, independent mirror toggles, and robot1 blend weight.")
        print("Opacity: click/drag, type a percentage + Enter, or use +/- and 0/50/100 presets.")

        def tick():
            try:
                for window in windows:
                    window.refresh()
            except Exception as exc:
                failure.append(exc)
                root.quit()
                return
            root.after(66, tick)

        root.after(0, tick)
        root.mainloop()
        if failure:
            raise failure[0]
    finally:
        for reader in started:
            reader.stop.set()
        for reader in started:
            reader.thread.join(timeout=6)
            if reader.thread.is_alive():
                print(f"Warning: ZED {reader.serial} is still shutting down.", file=sys.stderr)
        root.destroy()


def main(argv=None):
    args = parse_args(argv)
    try:
        dataset = DatasetReferences(args.dataset_root) if args.dataset_root is not None else None
        if dataset is not None:
            # Read/validate references before importing the camera SDK or opening devices.
            for pair in args.pairs:
                for robot in (0, 1):
                    ref = dataset.read(robot, pair.name.lower(), next(iter(dataset.episodes[robot])), 0)
                    print(f'{pair.name} robot{robot}: {ref.path} | frame 0 / {ref.count}')
        if args.check_reference:
            return 0
        if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
            raise RuntimeError("DISPLAY is not set. Run on the workstation's graphical desktop.")
        import pyzed.sl as sl

        cv2.setNumThreads(1)
        run(args, sl, dataset=dataset)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
