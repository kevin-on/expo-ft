#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside the existing Slurm allocation}"
[[ "$SLURM_JOB_ID" == 17539504 ]]
STAGE=/tmp/kevinon/expo-real-17539504
SNAPSHOT=/iris/u/kevinon/workspace/expo-ft-real/17539504
mkdir -p "$STAGE"/{home,runtime,output,cache,tmp,openpi-cache,demo,assets/validation-recording}
export APPTAINER_CACHEDIR="$STAGE/cache" APPTAINER_TMPDIR="$STAGE/tmp"
if [[ ! -f "$STAGE/learner.sif" ]]; then
 cp --reflink=auto --no-preserve=mode,ownership /iris/u/kevinon/artifacts/expo-ft/expo-ft-learner-jax053.sif "$STAGE/learner.sif"
fi
if [[ ! -f "$STAGE/weights-ready" ]]; then
 cp -R --reflink=auto --no-preserve=mode,ownership,xattr /iliad/u/kevinon/artifacts/expo-ft/openpi-cache/. "$STAGE/openpi-cache/"
 touch "$STAGE/weights-ready"
fi
cp -R --no-preserve=mode,ownership /iris/u/kevinon/artifacts/expo-ft/demo/. "$STAGE/demo/"
cp "$SNAPSHOT/norm_stats.json" "$STAGE/assets/validation-recording/norm_stats.json"
if [[ ! -d "$STAGE/source" ]]; then
 apptainer exec --cleanenv --containall --no-mount cwd --bind "$STAGE:/stage" "$STAGE/learner.sif" cp -a /opt/expo-ft /stage/source
fi
tar -xzf "$SNAPSHOT/source-overlay.tar.gz" -C "$STAGE/source"
cp "$SNAPSHOT/source-manifest.json" "$STAGE/"
cp "$SNAPSHOT/container.sh" "$STAGE/container.sh"
source "$STAGE/container.sh"
"${container[@]}" python docker/learner/verify_assets.py /openpi-cache > "$STAGE/output/assets-verified.txt" 2>&1
"${container[@]}" python -c 'import jax; import numpy as np; import train_pi_robo; from expo_ft.agents.alg.expo_ft import EXPOLearner; from expo_ft.utils.multi_robot_training import train_multi_robot; assert len(jax.devices()) == 2; print(jax.devices()); print("learner imports OK"); print(jax.jit(lambda x:x+1)(np.ones(2)).block_until_ready())' > "$STAGE/output/preflight.txt" 2>&1
"${container[@]}" python train_pi_robo.py --helpshort > "$STAGE/output/learner-help.txt" 2>&1 || test "$?" = 1
cp "$STAGE/output/preflight.txt" /iris/u/kevinon/outputs/expo-ft-real/17539504/preflight.txt
printf 'READY stage=%s\n' "$STAGE"
