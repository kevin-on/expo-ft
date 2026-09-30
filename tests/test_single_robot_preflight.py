"""Run the WS preflight with simulated enumeration, without importing device SDKs."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class PreflightTests(unittest.TestCase):
    def check_preflight(self, robot, missing_mouse=False):
        configs = {str(ROOT / f'configs/robots/robot-{i}.json'): json.loads(
            (ROOT / f'configs/robots/robot-{i}.json').read_text()) for i in (0, 1)}
        selected = configs[str(ROOT / f'configs/robots/robot-{robot}.json')]
        zed = ModuleType('pyzed.sl')
        zed.CAMERA_STATE = NS(AVAILABLE=1)
        zed.Camera = NS(get_device_list=lambda: [NS(serial_number=s, camera_state=1)
                                                for s in selected['camera_serials']])
        mouse = ModuleType('pyspacemouse')
        mouse.Enumeration = lambda: NS(find=lambda: [] if missing_mouse else [
            NS(path=selected['spacemouse_device_path'], product_string='SpaceMouse Wireless')])
        code = (ROOT / 'scripts/multi_robot/run_workstation_rollout.sh').read_text().split("<<'PY'\n")[1].split('\nPY\n')[0]
        read_text = Path.read_text
        def read(path, *a, **kw):
            if str(path).startswith('configs/robots/'):
                return json.dumps(configs[str(ROOT / path)])
            return read_text(path, *a, **kw)
        with patch.dict(sys.modules, {'pyzed': ModuleType('pyzed'), 'pyzed.sl': zed, 'pyspacemouse': mouse}), \
             patch.object(sys, 'argv', ['preflight', str(robot)]), patch.object(Path, 'read_text', read):
            exec(compile(code, '<mock-device-preflight>', 'exec'), {})

    def test_selected_robot_does_not_require_other_devices(self):
        for robot in (0, 1):
            with self.subTest(robot=robot):
                self.check_preflight(robot)

    def test_selected_robot_missing_mouse_still_fails(self):
        with self.assertRaisesRegex(SystemExit, 'Robot 0: configured SpaceMouse path missing'):
            self.check_preflight(0, missing_mouse=True)


if __name__ == '__main__':
    unittest.main()
