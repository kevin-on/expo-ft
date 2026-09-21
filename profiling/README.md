# Single-A40 update latency

Base commit: `f92c32a0d1eae3aca2282ff1787228c60b55a9be`

Profiling branch: `codex/profile-update-latency`

## Result

The unmodified learner runs warm updates in **1.163 seconds per call** on one
A40 with **batch size 2 and UTD 1**. The previous three-call smoke test measured
three different JIT compilations and therefore did not measure steady-state
training throughput.

Job 17534126 ran on iris6, A40
`GPU-b0df5ee7-1dac-1eb9-6561-a96f5d6a6eb4`, driver 610.43.02.

| Update | Original code (seconds) | Update executables cached |
| --- | ---: | ---: |
| 1 | 300.153 | 1 |
| 2 | 153.630 | 2 |
| 3 | 155.594 | 3 |
| 4 | 1.157 | 3 |
| 5 | 1.164 | 3 |
| 6 | 1.167 | 3 |

XLA compilation took 284.992, 145.060 and 146.825 seconds in the first
three calls. Calls 4–6 reused the third executable, with no new input-signature
changes. All six updates produced finite metrics, advanced the actor step,
changed actor/critic/edit-actor/encoder parameters, and completed a final
finite 16-by-7 action inference.

A separate baseline run, job 17533922 on the same node, reproduced the first
three slow calls (302.101, 156.092, 158.326 seconds). Raw timings and validation
outcomes are in [single-a40-results.json](single-a40-results.json).

The learner implementation in this branch remains identical to the base
commit. This investigation establishes warm throughput for the tested
configuration; it does not establish batch-64 or default-UTD performance.

## Cause of the misleading measurement

JAX tracing diagnostics identify two transient signature changes:

* Call 2: 18 target-critic parameter leaves gain mesh metadata, and the
  temperature parameter changes from weak to strong float32. The learner is
  passed as both `self` and `agent`, so JAX lists 38 input mismatches.
* Call 3: the temperature optimizer's Adam `mu` and `nu` change from weak to
  strong float32, giving four mismatches across the two learner arguments.

The state tree itself remains unchanged. The inference cache is already
removed before entering the compiled update.

## Startup optimization experiments

These are retained as **experimental patches**, not applied learner changes:

* [candidates/initialization.patch](candidates/initialization.patch) initializes
  the target critic from the already sharded critic parameters and gives the
  temperature an explicit float32 dtype. This candidate ran six calls on the
  same allocated GPU immediately after the baseline, with a separate cold JIT
  cache. Its times were 304.292, 153.472, 1.169, 1.159, 1.156, 1.166 seconds.
  It removed later tracing misses but still compiled a second executable.
  Initial and returned array sharding/commitment metadata must also be
  stabilized at the compilation boundary.
* The strict numerical comparison failed at the first actor gradient norm:
  baseline 1.21531212 versus candidate 1.21997726 (0.384% relative difference,
  tolerance `rtol=1e-4, atol=1e-5`). Both runs trained successfully, but this
  does **not** establish numerical equivalence. The difference's cause was not
  established, and the comparison stopped before testing final parameter and
  action equality. Job 17534126 therefore exited 1 in the comparison step.
* [candidates/normalize-single-device.patch](candidates/normalize-single-device.patch)
  additionally normalizes learner-array placement before a single-device
  update. A small CPU JAX example supported the cache-stabilization approach.
  This full candidate was **not GPU-tested** and is not an accepted fix.

Warm execution already takes about 1.16 seconds in the original code. No
unverified optimization is required to obtain that measured speed.

## Measurement method

`tests/gpu/learner_smoke.py --profile-updates` records JAX input types and
shardings, state-tree changes, update cache size and compiler logs. Timed calls
use `agent.update(...)` followed by `jax.block_until_ready((agent, info))`.
Batch preparation and validation hashes are outside the timed region.

The experiment uses the existing Docker-derived Apptainer image, public
`pi05_base` weights, initialized trainable parameters, seed 42, and recorded
robot observations. It retains action candidate sampling and actor, critic,
encoder, edit-actor and temperature updates. No robot is contacted.

The comparison tool deliberately uses strict numerical tolerances and requires
one cached update executable for the candidate. A failed comparison remains a
failed check; its tolerance was not loosened to accept this experiment.

## Reproduction and artifacts

The existing image supplies JAX/jaxlib 0.5.3, Flax 0.10.2 and Optax 0.2.4:

* Image: `/iris/u/kevinon/artifacts/expo-ft/expo-ft-learner-jax053.sif`
* SHA256: `3ca61c8fb26fab7597b01c22ca1e5599657a59ae155117010e0185ac11f2f7ef`
* Diagnostic results:
  `/iris/u/kevinon/outputs/expo-ft-update-profile/gpu-17533922/results.tar.gz`
* Six-call baseline and first candidate:
  `/iris/u/kevinon/outputs/expo-ft-update-profile/gpu-17534126/results.tar.gz`
* Exact submitted source snapshot:
  `/iris/u/kevinon/workspace/expo-ft-update-profile/comparison-v1`

`profiling/prepare_snapshot.py OUTPUT_DIR` captures the original commit and
the current learner files alongside the harness. On this branch the current
learner is also the original; apply a candidate explicitly in a separate
experimental checkout before snapshotting to reproduce that candidate.
The two candidate patches are alternatives, not cumulative patches.

Copy the small snapshot to the cluster and submit `run_profile.sbatch` from
it with an explicit `sbatch --chdir=...`. The script requests one A40, 8 CPUs,
64 GiB RAM and one hour; the two six-update processes run sequentially.
It stages image, weights, source, caches and detailed logs on local node storage,
then archives results to shared storage. Paths refer to this Stanford cluster
setup and need adjustment elsewhere.
