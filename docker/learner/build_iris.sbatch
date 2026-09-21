#!/bin/bash
# Stanford Iris/Iliad experiment script: account, partition and storage paths below
# are site-specific. Review them before reuse on a different GPU cluster.
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --job-name=expo-image-build
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --chdir=/iris/u/kevinon/workspace/expo-ft-docker-validation
#SBATCH --output=/iris/u/kevinon/outputs/expo-ft-docker-validation/logs/%x-%j.out
#SBATCH --error=/iris/u/kevinon/outputs/expo-ft-docker-validation/logs/%x-%j.err
set -euo pipefail
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/expo-build-${SLURM_JOB_ID}-XXXXXX")
ARTIFACTS=/iris/u/kevinon/artifacts/expo-ft
RESULTS=/iris/u/kevinon/outputs/expo-ft-docker-validation/build-${SLURM_JOB_ID}
mkdir -p "$STAGE/tmp" "$STAGE/cache" "$STAGE/source" "$RESULTS"
export TMPDIR="$STAGE/tmp" APPTAINER_TMPDIR="$STAGE/tmp" APPTAINER_CACHEDIR="$STAGE/cache"
recover() {
    rc=$?
    trap - EXIT
    cp "$STAGE/build.log" "$RESULTS/build.log" 2>/dev/null || true
    printf '%s\n' "$rc" > "$RESULTS/exit-code.txt"
    exit "$rc"
}
trap recover EXIT
hostname
apptainer --version
printf 'Local stage: %s\n' "$STAGE"
df -h "$STAGE"
tar -xzf source.tar.gz -C "$STAGE/source"
cp learner.def "$STAGE/learner.def"
cd "$STAGE"
apptainer build --fakeroot --mksquashfs-args "-processors ${SLURM_CPUS_PER_TASK}" \
    "$STAGE/expo-ft-learner-jax053.sif" learner.def > "$STAGE/build.log" 2>&1
apptainer exec --cleanenv "$STAGE/expo-ft-learner-jax053.sif" \
    cat /opt/expo-ft/docker/learner/requirements-installed.txt > "$RESULTS/requirements-installed.txt"
sha256sum "$STAGE/expo-ft-learner-jax053.sif" > "$RESULTS/image-sha256.txt"
cp "$STAGE/expo-ft-learner-jax053.sif" "$ARTIFACTS/expo-ft-learner-jax053.sif.partial"
mv "$ARTIFACTS/expo-ft-learner-jax053.sif.partial" "$ARTIFACTS/expo-ft-learner-jax053.sif"
printf 'Image ready: %s\n' "$ARTIFACTS/expo-ft-learner-jax053.sif"
