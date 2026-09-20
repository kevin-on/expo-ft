"""Verify EXPO-FT installs the exact integrated DROID fork without overwrites."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
DROID = ROOT / "client/droid"
SCRIPT = ROOT / "scripts/multi_robot/setup_droid.py"
spec = importlib.util.spec_from_file_location("setup_droid", SCRIPT)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def git(target, *args):
    return subprocess.check_output(["git", "-C", str(target), *args], text=True).strip()


@pytest.fixture
def checkout(tmp_path):
    return setup.install_checkout(tmp_path / "droid", source=str(DROID))


def test_local_install_pins_commit_and_is_idempotent(checkout):
    assert git(checkout, "rev-parse", "HEAD") == setup.REVISION
    assert not git(checkout, "status", "--porcelain")
    assert (checkout / "tests/test_multi_robot.py").exists()
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(checkout), "--source", str(DROID)],
            check=True, capture_output=True, text=True,
        )
        assert setup.REVISION in result.stdout
    assert not git(checkout, "status", "--porcelain")


def test_existing_local_configuration_is_preserved(checkout, capsys):
    readme = checkout / "README.md"
    content = readme.read_text() + "\nLocal setup notes\n"
    readme.write_text(content)
    marker = checkout / "local-config.txt"
    marker.write_text("local settings\n")
    setup.install_checkout(checkout, source="unused")
    assert readme.read_text() == content
    assert marker.read_text() == "local settings\n"
    assert "local changes preserved" in capsys.readouterr().out


def test_wrong_revision_is_refused_without_modification(checkout):
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "HEAD^"], check=True)
    before = git(checkout, "rev-parse", "HEAD")
    with pytest.raises(SystemExit, match="Use a separate checkout"):
        setup.install_checkout(checkout, source="unused")
    assert git(checkout, "rev-parse", "HEAD") == before
    assert not git(checkout, "status", "--porcelain")


def test_existing_nonrepository_and_parent_repository_are_refused(checkout, tmp_path):
    nonrepo = tmp_path / "existing"
    nonrepo.mkdir()
    marker = nonrepo / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(SystemExit, match="not a DROID checkout"):
        setup.install_checkout(nonrepo, source="unused")
    assert list(nonrepo.iterdir()) == [marker]
    assert marker.read_text() == "keep"

    child = checkout / "untracked-directory"
    child.mkdir()
    with pytest.raises(SystemExit, match="inside another repository"):
        setup.install_checkout(child, source="unused")
    assert not list(child.iterdir())
