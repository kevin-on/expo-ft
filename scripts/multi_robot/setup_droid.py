"""Clone the pinned DROID fork and apply this repository's multi-robot patch.

Run from either machine; an optional path selects the NUC's DROID checkout.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REVISION = "076cecd2c892e644fdc106f8ba3a79482ed6e0e8"


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "client/droid"
    patch = ROOT / "patches/droid-multi-robot.patch"
    if not target.exists():
        subprocess.run(["git", "clone", "https://github.com/pd-perry/droid.git", str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", REVISION], check=True)
    git = ["git", "-C", str(target)]
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"Expected DROID {REVISION}; found {revision}. Use a separate checkout.")
    if subprocess.run([*git, "apply", "--reverse", "--check", str(patch)], capture_output=True).returncode == 0:
        print("DROID multi-robot patch already applied")
        return
    subprocess.run([*git, "apply", "--check", str(patch)], check=True)
    subprocess.run([*git, "apply", str(patch)], check=True)
    print(f"Applied DROID multi-robot patch to {target}")


if __name__ == "__main__":
    main()
