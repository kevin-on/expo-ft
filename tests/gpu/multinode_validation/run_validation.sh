#!/usr/bin/env bash
# Runs after watch_allocation verifies job 3246733 and retires job 3237441.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ -z ${SLURM_JOB_ID:-} ]] || { echo 'Start from the login node'; exit 1; }
JOB=3246733
ASSET=expo_ft/pick_cube_balance_0923_mixed_20_seed3
mapfile -t NODES < <(python3 -c 'import json; print("\n".join(json.load(open("allocated-nodes.json"))["nodes"]))')
[[ ${#NODES[@]} == 2 ]] || exit 1
module load nccl-ofi-plugin/1.18.0-cuda129
COORDINATOR=${NODES[0]}:29451
RESOURCES=(--jobid="$JOB" --overlap --exact --ntasks-per-node=1 --cpus-per-task=48
           --gpus-per-node=4 --mem=440G --kill-on-bad-exit=1
           --network=single_node_vni,disable_rdzv_get)
squeue --steps -j "$JOB"
trap 'status=$?; printf "%s\n" "$status" > driver-exit-code; exit "$status"' EXIT
if [[ ${1:-} != --updates-only ]]; then
printf 'stage\n' > phase
srun "${RESOURCES[@]}" -N2 -n2 --time=00:30:00 python3 "$PWD/compute.py" \
    --phase stage --asset-id "$ASSET" 2>&1 | tee stage.log
fi
if [[ ${1:-} != --updates-only && ${1:-} != --skip-collective && ${1:-} != --blocked ]]; then
printf '8gpu-collective\n' > phase
srun "${RESOURCES[@]}" -N2 -n2 --time=00:15:00 python3 "$PWD/compute.py" \
    --phase collective --processes 2 --coordinator "$COORDINATOR" --asset-id "$ASSET" \
    2>&1 | tee 8gpu-collective-launch.log
# The process also forces NCCL_NET=AWS Libfabric; keep actual CXI evidence.
grep -iE 'provider.*cxi|cxi.*provider' 8gpu-collective/nccl.*.log > cxi-evidence.txt
fi
EXTRA=()
SUFFIX=
if [[ ${1:-} == --blocked ]]; then
    EXTRA+=(--utd-axis)
    SUFFIX=-blocked
fi
for COUNT in 1 2; do
    printf '%sgpu-update\n' "$((COUNT * 4))" > phase
    srun "${RESOURCES[@]}" -N"$COUNT" -n"$COUNT" --nodelist="$(IFS=,; echo "${NODES[*]:0:$COUNT}")" \
        --time=01:30:00 python3 "$PWD/compute.py" --phase update --processes "$COUNT" \
        --coordinator "$COORDINATOR" --asset-id "$ASSET" "${EXTRA[@]}" \
        2>&1 | tee "$((COUNT * 4))gpu-update${SUFFIX}-launch.log"
done
printf 'update-benchmarks-complete-awaiting-review\n' > phase
touch UPDATE_BENCHMARKS_DONE
