"""Exercise the patched hardware routing with SDK/process calls replaced."""
import ast
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
DROID = ROOT / "client/droid"


def load_class(relative_path, name, namespace):
    path = DROID / relative_path
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    namespace["__file__"] = str(path)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_controller_endpoints_and_scoped_process_shutdown():
    import os
    processes, commands, connections = [], [], []
    class Process:
        def __init__(self, command, **kwargs):
            assert kwargs["start_new_session"]
            self.pid = 100 + len(processes)
            self.stdin = io.StringIO()
            processes.append((self, command))
        def poll(self):
            return None
        def wait(self, timeout):
            pass
    def interface(**kwargs):
        connections.append(kwargs)
        return SimpleNamespace(metadata=SimpleNamespace(max_width=1))
    robot_type = load_class("droid/franka/robot.py", "FrankaRobot", dict(
        os=os, subprocess=SimpleNamespace(Popen=Process, PIPE=-1, run=lambda command, **_: commands.append(command)),
        time=SimpleNamespace(sleep=lambda _: None), sudo_password="", RobotInterface=interface,
        GripperInterface=interface, RobotIKSolver=lambda: None,
    ))
    first = robot_type("172.16.0.2", 50051, 50052, "/dev/serial/by-id/gripper-A")
    second = robot_type("172.16.0.3", 50061, 50062, "/dev/serial/by-id/gripper-B")
    first.launch_controller()
    second.launch_controller()
    first.launch_robot()
    second.launch_robot()
    assert [call["port"] for call in connections] == [50051, 50052, 50061, 50062]
    assert processes[2][1][-2:] == ["172.16.0.3", "50061"]
    assert processes[3][1][-2:] == ["/dev/serial/by-id/gripper-B", "50062"]
    second.kill_controller()
    assert [command[-1] for command in commands] == ["-102", "-103"]
    assert len(first._controller_processes) == 2


def test_rpc_port_and_attach_without_controller_restart():
    endpoints, launches = [], []
    rpc = SimpleNamespace(connect=endpoints.append, launch_robot=lambda: launches.append("attach"))
    cls = load_class("droid/misc/server_interface.py", "ServerInterface", dict(
        zerorpc=SimpleNamespace(Client=lambda **_: rpc),
    ))
    cls(ip_address="172.16.0.1", port=4243, launch=False)
    assert endpoints == ["tcp://172.16.0.1:4243"]
    assert launches == ["attach"]


def test_camera_filter_before_opening():
    path = DROID / "droid/camera_utils/camera_readers/zed_camera.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "gather_zed_cameras")
    opened = []
    namespace = dict(
        sl=SimpleNamespace(Camera=SimpleNamespace(get_device_list=lambda: [SimpleNamespace(serial_number=i) for i in (11, 12, 21, 22)])),
        ZedCamera=lambda camera, wrist: opened.append((camera.serial_number, wrist)),
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    namespace["gather_zed_cameras"](["21", "22"], "22")
    assert opened == [(21, "22"), (22, "22")]


def test_patch_applies_cleanly_and_setup_is_idempotent(tmp_path):
    import tarfile
    archive = subprocess.check_output(["git", "-C", str(DROID), "archive", "HEAD"])
    checkout = tmp_path / "droid"
    checkout.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(checkout, filter="data")
    patch = ROOT / "patches/droid-multi-robot.patch"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=checkout, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=checkout, check=True)
    subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=checkout, check=True)
    for _ in range(2):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/multi_robot/setup_droid.py")],
                                check=True, capture_output=True, text=True)
        assert "already applied" in result.stdout
    for script in ("launch_robot.sh", "launch_gripper.sh"):
        path = DROID / "droid/franka" / script
        assert "pkill" not in path.read_text()
        subprocess.run(["bash", "-n", str(path)], check=True)
