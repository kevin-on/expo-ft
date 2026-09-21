#!/bin/bash
#SBATCH --account=bgqe-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=expo-h200-validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=250G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
# Submit from the repository root; create logs/ before sbatch opens its files.
cd "$SLURM_SUBMIT_DIR"
: "${EXPO_OUTPUT_ROOT:?Set a persistent parent directory for job outputs}"
export EXPO_OUTPUT_DIR="$EXPO_OUTPUT_ROOT/$SLURM_JOB_NAME-$SLURM_JOB_ID"
if [[ $# == 0 ]]; then set -- validate; fi
srun bash docker/learner/delta/run.sh "$@"
