"""Stage and run one validation rank inside the requested GH200 allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
JOB = "3246733"
STAGE = Path("/tmp/expo-gh200x8-" + JOB)
IMAGE = Path("/projects/bgqe/kon/expo-ft/access-runtime/expo-sft-learner-jax053-aarch64-20260923.sif")
CHECKPOINT = Path("/work/hdd/bgqe/kon/expo-ft/sft-runs/balance0923-seed3-5k-20260923/runs/mixed-020/checkpoints/expo_pi05_droid_lora_finetune_sft_cartesian_state/mixed-020-seed3-b64-s42-5k-deltaai-20260923/4999")
FIXTURE = Path("/work/hdd/bgqe/kon/expo-ft/split-validation/20260927-xxh3-batch-e2e/fixture")
TOKENIZER = Path("/projects/bgqe/kon/expo-ft/baseline-20260921/openpi-cache/big_vision/paligemma_tokenizer.model")


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def stage(fabric=True):
    STAGE.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    source = STAGE / ("source-" + manifest["source.tar"][:12])
    for name in ("source.tar", "openpi.tar"):
        if digest(ROOT / name) != manifest[name]:
            raise ValueError("Source archive digest mismatch: " + name)
    if not source.exists():
        source.mkdir()
        with tarfile.open(ROOT / "source.tar") as archive:
            archive.extractall(source)
        op = source / "expo_ft/agents/vla/openpi"
        op.mkdir(parents=True, exist_ok=True)
        with tarfile.open(ROOT / "openpi.tar") as archive:
            archive.extractall(op)
    ready = STAGE / "inputs-ready.json"
    if not ready.exists():
        if shutil.disk_usage(STAGE).free < 40 * 1024**3:
            raise RuntimeError("Need at least 40 GiB of node-local space")
        for name, original in (("runtime.sif", IMAGE),):
            temporary = STAGE / (name + ".partial")
            shutil.copyfile(original, temporary)
            temporary.replace(STAGE / name)
        run("rsync", "-a", str(CHECKPOINT) + "/", str(STAGE / "checkpoint") + "/")
        run("rsync", "-a", str(FIXTURE) + "/", str(STAGE / "fixture") + "/")
        (STAGE / "model-cache/big_vision").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TOKENIZER, STAGE / "model-cache/big_vision/paligemma_tokenizer.model")
        ready.write_text(json.dumps(dict(checkpoint=str(CHECKPOINT), fixture=str(FIXTURE),
                                        image=str(IMAGE), source=manifest), indent=2))

    if not fabric:
        print("STAGED", os.uname().nodename, str(STAGE), flush=True)
        return
    # Expose the site's OFI plugin and its dependency closure to the existing
    # container. Keep the container's glibc and C++ runtime; do not replace its
    # Python/JAX/NCCL installation or edit the shared SIF.
    plugin = Path(os.environ["NCCL_NET_PLUGIN"])
    libraries = subprocess.check_output(["ldd", str(plugin)], universal_newlines=True)
    if "not found" in libraries:
        raise RuntimeError("Site OFI plugin has an unresolved host dependency")
    destination = STAGE / "fabric-libs"
    destination.mkdir(exist_ok=True)
    paths = [plugin] + [Path(value) for value in re.findall(r"=> (/\S+)", libraries)]
    skip = re.compile(r"^(lib(c|m|mvec|pthread|rt|dl|resolv|anl|util|stdc\+\+|gcc_s|crypto|ssl)\.so|ld-linux)")
    # Python's SSL extension requires the image's newer OpenSSL. The OFI
    # dependencies use its backward-compatible symbols too; do not shadow it.
    for path in destination.glob('lib*.so*'):
        if skip.match(path.name):
            path.unlink()
    for path in paths:
        if skip.match(path.name) or path.name.startswith("libcuda.so"):
            continue
        target = destination / path.name
        if not target.exists():
            shutil.copyfile(path, target)
    (STAGE / "fabric-dependencies.txt").write_text(libraries)
    print("STAGED", os.uname().nodename, str(STAGE), flush=True)


def execute(args, command=None, output_name=None, extra_bindings=(), fabric=True):
    rank = int(os.environ["SLURM_PROCID"])
    output = ROOT / (output_name or (str(args.processes * 4) + "gpu-" + args.phase + ("-blocked" if args.utd_axis else "")))
    output.mkdir(exist_ok=True)
    local = STAGE / (output.name + "-" + os.environ["SLURM_STEP_ID"])
    for name in ("home", "runtime", "cache"):
        (local / name).mkdir(parents=True, exist_ok=True)
    # JAX writes distributed compilation entries only from rank zero. Every
    # rank must read that same cache on restart, not a different local /tmp.
    jax_cache = ROOT / 'jax-cache-global8' if args.processes > 1 else STAGE / 'jax-cache'
    jax_cache.mkdir(exist_ok=True)
    settings = dict(CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
        JAX_PLATFORMS="cuda", XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION=".90", OPENPI_DATA_HOME="/model-cache",
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", XDG_CACHE_HOME="/cache",
        TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", FASTVLA_GATHER_THREADS="12", WANDB_MODE="disabled",
        EXPO_FSDP_DEVICES="1", JAX_COMPILATION_CACHE_DIR="/jax-cache",
        EXPO_PROCESS_COUNT=str(args.processes), EXPO_PROCESS_ID=str(rank),
        EXPO_COORDINATOR=args.coordinator,
        PYTHONPATH="/opt/expo-ft:/opt/expo-ft/expo_ft/agents/vla/openpi/src:/opt/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client/src")
    if args.phase == 'cpu':
        settings['JAX_PLATFORMS'] = 'cpu'
        # Test tooling is separate from the immutable training image and is
        # reused across CPU checks in this allocation. Never expose it to GPU runs.
        test_deps = STAGE / 'test-deps'
        test_deps.mkdir(exist_ok=True)
        extra_bindings = [*extra_bindings, str(test_deps) + ':/test-deps']
        settings['PYTHONPATH'] = '/test-deps:' + settings['PYTHONPATH']
        settings['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
        command = ['python', '-B', '-c',
            'import importlib.util, shutil, subprocess, sys, unittest\n'
            'if importlib.util.find_spec("pytest") is None:\n'
            '    assert shutil.disk_usage("/test-deps").free > 100 * 1024**2, "Need 100 MiB for test tools"\n'
            '    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", '
            '"--no-cache-dir", "--target", "/test-deps", "pytest==8.3.5"], check=True)\n'
            'loader=unittest.TestLoader(); '
            'suite=unittest.TestSuite(loader.discover("tests/distributed", pattern=p) '
            'for p in ("test_runner.py", "test_policy.py", "test_learner_group.py")); '
            'result=unittest.TextTestRunner(verbosity=2).run(suite); '
            'assert result.wasSuccessful(); '
            'subprocess.run(["python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", '
            '"tests/cpu/test_distributed_sampler.py", "tests/cpu/test_robot_replay.py", '
            '"tests/cpu/test_replay_persistence.py"], check=True)']
    if fabric:
        for key, value in os.environ.items():
            if key.startswith(("NCCL_", "FI_", "SLINGSHOT_")):
                settings[key] = value
        settings.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET",
                    NCCL_DEBUG_FILE="/output/nccl.%h.%p.log",
                    NCCL_NET_PLUGIN="/fabric-libs/libnccl-net.so",
                    LD_LIBRARY_PATH="/fabric-libs:/.singularity.d/libs")
    if args.processes > 1:
        # Fail explicitly instead of timing an unnoticed TCP fallback.
        settings["NCCL_NET"] = "AWS Libfabric"
    env = dict(os.environ, **{"APPTAINERENV_" + k: v for k, v in settings.items()})
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    source = STAGE / ("source-" + manifest["source.tar"][:12])
    bindings = [str(source) + ":/opt/expo-ft:ro",
                str(STAGE / "checkpoint") + ":/checkpoint:ro",
                str(STAGE / "fixture") + ":/fixture:ro",
                str(STAGE / "model-cache") + ":/model-cache",
                str(output) + ":/output", str(local / "cache") + ":/cache",
                str(jax_cache) + ":/jax-cache"]
    if fabric:
        bindings.append(str(STAGE / "fabric-libs") + ":/fabric-libs:ro")
    bindings.extend(extra_bindings)
    cxi_devices = sorted(Path("/dev").glob("cxi[0-9]*"))
    if args.processes > 1 and not cxi_devices:
        raise RuntimeError("No Slingshot CXI devices exposed by the allocation")
    bindings.extend(str(path) + ":" + str(path) for path in cxi_devices)
    container = ["apptainer", "exec", "--nv", "--cleanenv", "--containall", "--no-mount", "cwd",
                 "--home", str(local / "home") + ":/home/kon", "--workdir", str(local / "runtime"),
                 "--pwd", "/opt/expo-ft", "--bind", ",".join(bindings), str(STAGE / "runtime.sif")]
    if fabric:
        run(*container, "python", "-c", "import ctypes,os,ssl; ctypes.CDLL(os.environ['NCCL_NET_PLUGIN']); print('OFI_PLUGIN_AND_SSL_LOAD_OK', ssl.OPENSSL_VERSION)", env=env)
    if command is not None:
        run(*container, *command, env=env)
        return
    run(*container, "python", "-u", "tests/gpu/multinode_update.py", "--phase", args.phase,
        "--output", "/output", "--fixture", "/fixture", "--params", "/checkpoint/params",
        "--assets", "/checkpoint/assets", "--asset-id", args.asset_id,
        *(["--utd-axis"] if args.utd_axis else []), env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("stage", "collective", "control", "update", "cpu"), required=True)
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--coordinator", default="")
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--utd-axis", action="store_true")
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_ID") != JOB or os.uname().machine != "aarch64":
        raise RuntimeError("Run only inside the requested GH200 allocation")
    if args.phase == "stage":
        stage()
    else:
        execute(args)


if __name__ == "__main__":
    main()
