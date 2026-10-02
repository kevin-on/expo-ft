# Test inventory

Use this inventory instead of dated allocation scripts. Do not run models,
large conversion tests, or full CPU/JAX suites on a workstation being used for
robot experiments. Use an allocated compute node for substantial tests. No test
is authorized to access physical hardware merely because it is named a test.

## Regression suites to retain

| Location | Purpose / dependencies |
| --- | --- |
| `cpu/` | Replay mixing and restore, multi-robot barriers/RPC, mirror/eval/config and conversion; Python3.11, `cpu/requirements.txt`, openpi-client; CPU JAX |
| `cpu/test_distributed_sampler.py` | Replacement sampling, small/empty pools, rank slices, success filtering, post-insertion batch timing and replay-restore reproduction |
| `distributed/` `test_*.py` | RAM transport, TLS, corruption/failover, relay supervision, policy contract, cancellation and checkpoint resume; standard unittest, xxhash/numpy/websockets as used by each test |
| `../client/tests/` | Mocked HID/camera/RPC, teleop vertical/bounds/reset, collection persistence/video; client test dependencies, no device access |
| `test_async_video_isolated.py` | Video ownership/backpressure/drain, terminal response before encoding completes |
| `test_split_reset_overlap_isolated.py` | Reset/update join, failure cleanup, durability before reset permission |
| `test_local_reset_overlap_isolated.py` | Equivalent local-mode reset/update ordering and shutdown |

The isolated tests are regression coverage for current behavior, not disposable
implementation experiments. Preserve them when cleaning up measurement scripts.
Their fakes avoid robot/GPU imports; they do not replace real-model verification.

`test_rollout_dashboard_isolated.py` covers resumed counters and live transition
display. `cpu/test_robot_round.py` checks handoff exclusion, including terminal
handoffs, without dropping intentional zero actions. `distributed/test_runner.py`
checks old/new checkpoint progress restore and handoff transport/replay cursors.

`test_colocated_isolated.py` runs the split round/barrier/failure regressions with
the actual in-process mailbox and verifies supervisor shutdown. It is stdlib-only.
`cpu/test_local_policy.py` uses tiny JAX arrays to check shared buffer pointers,
one/two-device replica selection, refresh after a donated update, and independent
inference RNGs. It also runs both actual split runners and the collector with a
tiny fake model/robot, including batch replay persistence and resumed inference
RNGs for one/two robots and one/two virtual devices. It uses no model weights or
hardware; expose two virtual CPU
devices for full coverage:

```bash
python tests/test_colocated_isolated.py
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  python -m pytest -q tests/cpu/test_local_policy.py
```

These checks establish the local delivery/parameter-sharing semantics, not GPU
throughput or full-model GPU execution.

In the matching test environment, on an authorized idle compute node:

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/cpu
python -m unittest discover -s tests/distributed -p 'test_*.py'
python tests/test_async_video_isolated.py
python tests/test_split_reset_overlap_isolated.py
python tests/test_local_reset_overlap_isolated.py
# In the client environment, not an environment lacking its dependencies:
python -m unittest discover -s client/tests -p 'test_*.py'
```

The `cpu/conftest.py` stand-ins are scoped to that pytest suite. Do not combine
unrelated suites in one Python process or install missing dependencies into a
live robot environment. Test only what a change affects; syntax/link checks
suffice for documentation-only edits.

For the standalone SpaceMouse/override tests on WS, reuse an isolated pytest
install (no changes to the active client environment). `--noconftest` avoids
the unrelated replay/JAX fixtures; this file supplies its own simulated HID:

```bash
source /scr/kevinon/env.sh
uv pip install --python client/.venv/bin/python --target /scr/kevinon/tmp/expo-hil-test-deps pytest
PYTHONPATH=/scr/kevinon/tmp/expo-hil-test-deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 client/.venv/bin/python -m pytest --noconftest \
  tests/cpu/test_spacemouse.py -q -p no:cacheprovider
```

## Reusable integration / measurement tools

- `test_remote_eval.py` and `test_robot_eval.py`: RAM envelope persistence,
  receive/eval exclusion, transport admission and independent robot starts.
- `gpu/remote_eval.py`: separate full-checkpoint and RAM-only GPU inference
  comparisons. `gpu/remote_eval_lifecycle.py` uses an actual payload and disposable
  GPU worker with fake robot clients to verify receive/eval/save transitions.
  See [remote eval](../docs/remote_checkpoint_eval.md).

- `gpu/learner_smoke.py`: a recorded HDF5 episode, real inference/updates and
  an independent-process checkpoint restore on one learner deployment.
- `gpu/split_smoke.py`: two model/transport processes on one allocated test node;
  seeded action parity, reset barrier, policy refresh and checkpoint save/restore.
  Useful when no WAN route is available; it does not measure WAN latency.
- `gpu/colocated_smoke.py`: actual `train_pi_robo.py --split_role=colocated`
  on two allocated GPUs, using recorded trajectories in place of robot RPCs.
  Checks GPU0 parameter buffer sharing, seeded action equality with the split
  snapshot path, real gradient updates, exact replay records and checkpoint
  resume. A three-round run with `--num-robot 1` followed by the same output with
  `--resume --rounds 1` covers persistence. Check the other robot count separately.
  Once persistence/parity is established, `--rounds-only` skips model checkpoint
  saves and snapshot comparisons while retaining updates, sharing and replay checks;
  use `--batch-size 64 --utd-ratio 20 --num-updates 3` for production update timing.
  W&B should be disabled. Its validation-only snapshot comparison is excluded
  from production delivery timings. No WS or device connections are opened.
- `gpu/split_wan_smoke.py`: two machines, recorded trajectories, real updates and
  loopback mock robot clients on inference. `--playback-hz=10` reproduces collection
  cadence; default 1000 is accelerated. See [the operating guide](../scripts/split/README.md#6-robot-free-10hz-verification)
  for fixtures, container bindings and the five-cycle timing procedure.
- `distributed/benchmark_transport.py`: isolate RAM/TLS transport throughput and
  verification from model execution. Only run with an isolated test mailbox and
  free endpoints; do not attach it to a live training sidecar.
- `gpu/multinode_update.py`: actual 4/8-GPU collective/control and fixed-global-
  batch update checks. `gpu/multinode_validation/` contains the dated allocation
  launch recipe. See [two-node learner](../docs/multinode_learner.md) for ownership,
  batch semantics, environment and full-iteration verification.

Pass explicit inputs/output directories and inspect each tool's `--help` on the
appropriate compute node. GPU tests must not be launched by routine test discovery.
Parameter transfer equality is distinct from cross-platform action identity.

## Removed historical harnesses

The fixed 2026-09-20/21 Iris/A40/H200 Slurm experiments, their speed analyzers,
the job 17539504 deployment snapshot, and the TPUQ smoke harness pinned to the
old incompatible OpenPI version were removed. Git history preserves them.
Current GPU and CPU regression coverage remains. TPU support in application
code is not removed; it would need a newly validated environment/launcher.

Retained evidence, with distinct scopes:

- One-node fork deployment: `/scr/kevinon/workspace/expo-ft-validation/20260927-fork-10hz-5/REPORT.md`.
- Matched four/eight-GPU comparison: `/scr/kevinon/workspace/expo-ft-validation/20260928-gh200x8/REPORT.md`.
- Distributed sampler and final eight-GPU run: `/scr/kevinon/workspace/expo-ft-validation/20260928-distributed-sampler/REPORT.md`.

Keep these bundles and live relay auxiliary files; they are not disposable logs.

- `test_model_config.py`: SFT/EXPO metadata, camera omission, override drift, resume and normalization checks (CPU in the full learner environment).
- The model-config and remote-eval suites also cover compact SFT initialization,
  base/schema/config mismatch rejection, packet-only normalization, full/compact
  metadata relocation, and new/legacy weight filenames.
- `gpu/trainable_checkpoint.py`: real full/compact SFT online initialization,
  exact initial parameter hashes, inference and one-update parity, then an
  independent online checkpoint restore. Run `export` on CPU and `full`, `compact`,
  `restore` in separate GPU processes. Use node-local `--output` scratch for the
  test weights/checkpoint and persistent `--reports` for validation evidence.
  Parameters, transformed inputs and RNG must match exactly. If cross-process
  GPU action comparisons need a larger `--action-atol`, first measure the same
  full checkpoint twice (`full --inference-only` in separate output directories);
  retain that repeat-control difference alongside the comparison results.
- `gpu/checkpoint_config.py`: real SFT inference, one online update and independent checkpoint restore/action parity, using recorded observations only.

If the read-only compute image lacks pytest, install it into node-local scratch:
`python -m pip install --target /cache/test-deps pytest`, then prepend
`/cache/test-deps` to `PYTHONPATH` for test commands only. Do not rebuild the shared
runtime or alter a running robot's environment for test dependencies.

- `test_robot_eval.py`: two-robot Space gating, first-plan barrier, independent reset,
  mirrored inputs/actions and shared-model/failure behavior; fake environments only.
- `gpu/robot_eval.py`: one coordinator round with recorded observations and a real
  SFT or online policy on one allocated GPU. No WS or hardware connections.

Run `python tests/cpu/test_sft_eval.py` separately: its import-isolation assertion
intentionally requires a process that has not imported JAX. Do not combine that
assertion with RPC tests that import the training stack.

- Camera runtime, WS frame conversion and timestamp propagation checks: see [camera observation runtime](../docs/camera_observation_runtime.md). These use fake SDK/RPC interfaces and require the matching DROID source on PYTHONPATH; they never open cameras or robots.
