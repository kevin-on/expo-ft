"""Install the pinned DROID fork without modifying an existing checkout.

Use --source /path/to/local/droid for offline or unpublished local integration.
The optional target also supports a separate checkout for NUC deployment.
"""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "https://github.com/kevin-on/droid.git"
REVISION = "5f9df37cf10153868a2f05eab43d6c32a68801bc"


def install_checkout(target, source=REPOSITORY):
    target = Path(target).resolve()
    if not target.exists():
        subprocess.run(["git", "clone", "--no-checkout", source, str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", REVISION], check=True)

    git = ["git", "-C", str(target)]
    try:
        root = subprocess.check_output(
            [*git, "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.PIPE
        ).strip()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"{target} is not a DROID checkout. Use a separate empty path.") from error
    if Path(root).resolve() != target:
        raise SystemExit(f"{target} is inside another repository. Use a separate checkout.")

    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"Expected DROID {REVISION}; found {revision}. Use a separate checkout.")
    dirty = subprocess.check_output(
        [*git, "status", "--porcelain", "--untracked-files=normal"], text=True
    ).strip()
    suffix = " (local changes preserved)" if dirty else ""
    print(f"DROID {REVISION} ready at {target}{suffix}")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", type=Path, default=ROOT / "client/droid")
    parser.add_argument("--source", default=REPOSITORY, help="Git URL or local repository to clone.")
    args = parser.parse_args(argv)
    install_checkout(args.target, args.source)


if __name__ == "__main__":
    main()
