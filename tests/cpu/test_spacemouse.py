"""Use PySpaceMouse 1.1.5's actual selection/decoder code with simulated HID I/O."""
import importlib.metadata
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def mouse_modules(monkeypatch):
    devices, opened, reads = [], [], []
    class HID:
        def __init__(self, path, vid, pid):
            self.path, self.vendor_id, self.product_id = path, vid, pid
            self.product_string = "test mouse"
            self.manufacturer_string = "test"
            self.release_number = 1
            self.serial_number = ""
        def open(self):
            opened.append(self.path)
        def set_nonblocking(self, value):
            pass
        def read(self, size):
            reads.append(self.path)
            return []
    monkeypatch.setitem(sys.modules, "easyhid", SimpleNamespace(
        Enumeration=lambda: SimpleNamespace(find=lambda: [HID(*device) for device in devices]),
        HIDException=RuntimeError,
    ))

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    sdk_path = importlib.metadata.distribution("pyspacemouse").locate_file("pyspacemouse/pyspacemouse.py")
    sdk = load("_cpu_spacemouse_sdk", sdk_path)
    monkeypatch.setitem(sys.modules, "pyspacemouse", sdk)
    root = Path(__file__).resolve().parents[2]
    wrapper = load("_cpu_spacemouse_wrapper", root / "client/real_utils/spacemouse.py")
    # Suppress only the background reader; tests explicitly run a read iteration.
    monkeypatch.setattr(wrapper, "threading", SimpleNamespace(
        Lock=threading.Lock, Thread=lambda **_: SimpleNamespace(start=lambda: None),
    ))
    return sdk, wrapper, devices, opened, reads


@pytest.mark.parametrize("names", [
    ("SpaceNavigator", "SpaceMouse Compact"), ("SpaceNavigator", "SpaceNavigator"),
])
def test_exact_paths_select_device_and_correct_decoder(mouse_modules, monkeypatch, names):
    sdk, wrapper, devices, opened, reads = mouse_modules
    for index, name in enumerate(names):
        devices.append((f"/hid/mouse-{index}", *sdk.device_specs[name].hid_id))
    policies = [wrapper.SpaceMousePolicy(device_path=f"/hid/mouse-{index}", device_number=99)
                for index in range(2)]
    assert opened == ["/hid/mouse-0", "/hid/mouse-1"]
    for index, policy in enumerate(policies):
        assert policy.spacemouse.device.name == sdk.device_specs[names[index]].name
        assert policy.spacemouse.device.device.path == f"/hid/mouse-{index}"
    # Opening the second device changes the SDK global. Each reader must still
    # use its own handle, rather than read from the last-opened device.
    def finish_read(_):
        raise InterruptedError("one iteration complete")
    monkeypatch.setattr(wrapper, "time", SimpleNamespace(sleep=finish_read))
    for policy in policies:
        with pytest.raises(InterruptedError):
            policy.spacemouse._read_spacemouse()
    assert reads == ["/hid/mouse-0", "/hid/mouse-1"]


def test_missing_or_unsupported_path_never_opens_another_mouse(mouse_modules):
    sdk, wrapper, devices, opened, _ = mouse_modules
    devices.append(("/hid/mouse-0", *sdk.device_specs["SpaceNavigator"].hid_id))
    devices.append(("/hid/unsupported", 0xFFFF, 0xFFFF))
    with pytest.raises(ValueError, match="path not found"):
        wrapper.SpaceMouseExpert(device_path="/hid/missing")
    with pytest.raises(ValueError, match="Unsupported SpaceMouse"):
        wrapper.SpaceMouseExpert(device_path="/hid/unsupported")
    assert opened == []


def test_legacy_same_model_number_selection(mouse_modules):
    sdk, wrapper, devices, opened, _ = mouse_modules
    for index in range(2):
        devices.append((f"/hid/mouse-{index}", *sdk.device_specs["SpaceNavigator"].hid_id))
    wrapper.SpaceMouseExpert(device_number=0)
    wrapper.SpaceMouseExpert(device_number=1)
    assert opened == ["/hid/mouse-0", "/hid/mouse-1"]


def test_rollout_forwards_hid_path(mouse_modules, monkeypatch):
    from client import run_client
    sdk, wrapper, devices, opened, _ = mouse_modules
    devices.extend([
        ("/hid/mouse-0", *sdk.device_specs["SpaceNavigator"].hid_id),
        ("/hid/mouse-1", *sdk.device_specs["SpaceMouse Compact"].hid_id),
    ])
    monkeypatch.setitem(sys.modules, "client.real_utils.spacemouse", wrapper)
    monkeypatch.setattr(run_client, "_spacemouse_policy", None)
    run_client._get_human_override_action(SimpleNamespace(
        collect_max_lin_vel=0.5, collect_max_rot_vel=0.1, spacemouse_device_path="/hid/mouse-1",
    ))
    assert opened == ["/hid/mouse-1"]
    assert run_client._spacemouse_policy.spacemouse.device.name == sdk.device_specs["SpaceMouse Compact"].name
