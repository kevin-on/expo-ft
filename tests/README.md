# Test inventory

Use this inventory instead of dated allocation scripts. Do not run models,
large conversion tests, or full CPU/JAX suites on a workstation being used for
robot experiments. Use an allocated compute node for substantial tests. No test
is authorized to access physical hardware merely because it is named a test.

## Regression suites to retain

| Location | Purpose / dependencies |
| --- | --- |
| `cpu/` | Replay mixing and restore, multi-robot barriers/RPC, mirror/eval/config and conversion; Python3.11, `cpu/requirements.txt`, openpi-client; CPU JAX |
| `distributed/` `test_*.py` | RAM transport, TLS, corruption/failover, relay supervision, policy contract, cancellation and checkpoint resume; standard unittest, xxhash/numpy/websockets as used by each test |
| `../client/tests/` | Mocked HID/camera/RPC, teleop vertical/bounds/reset, collection persistence/video; client test dependencies, no device access |
| `test_async_video_isolated.py` | Video ownership/backpressure/drain, terminal response before encoding completes |
| `test_split_reset_overlap_isolated.py` | Reset/update join, failure cleanup, durability before reset permission |
| `test_local_reset_overlap_isolated.py` | Equivalent local-mode reset/update ordering and shutdown |

The isolated tests are regression coverage for current behavior, not disposable
implementation experiments. Preserve them when cleaning up measurement scripts.
Their fakes avoid robot/GPU imports; they do not replace real-model verification.

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

## Reusable integration / measurement tools

- `gpu/learner_smoke.py`: a recorded HDF5 episode, real inference/updates and
  an independent-process checkpoint restore on one learner deployment.
- `gpu/split_smoke.py`: two model/transport processes on one allocated test node;
  seeded action parity, reset barrier, policy refresh and checkpoint save/restore.
  Useful when no WAN route is available; it does not measure WAN latency.
- `gpu/split_wan_smoke.py`: two machines, recorded trajectories, real updates and
  loopback mock robot clients on inference. `--playback-hz=10` reproduces collection
  cadence; default 1000 is accelerated. See [the operating guide](../scripts/split/README.md#6-robot-free-10hz-verification)
  for fixtures, container bindings and the five-cycle timing procedure.
- `distributed/benchmark_transport.py`: isolate RAM/TLS transport throughput and
  verification from model execution. Only run with an isolated test mailbox and
  free endpoints; do not attach it to a live training sidecar.

Pass explicit inputs/output directories and inspect each tool's `--help` on the
appropriate compute node. GPU tests must not be launched by routine test discovery.
Parameter transfer equality is distinct from cross-platform action identity.

## Removed historical harnesses

The fixed 2026-09-20/21 Iris/A40/H200 Slurm experiments, their speed analyzers,
the job 17539504 deployment snapshot, and the TPUQ smoke harness pinned to the
old incompatible OpenPI version were removed. Git history preserves them.
Current GPU and CPU regression coverage remains. TPU support in application
code is not removed; it would need a newly validated environment/launcher.

Latest evidence is at
`/scr/kevinon/workspace/expo-ft-validation/20260927-fork-10hz-5/REPORT.md`.
Keep that bundle and live relay auxiliary files; they are not disposable logs.
