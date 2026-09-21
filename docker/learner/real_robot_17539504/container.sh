# Source from a job step. All Python libraries and runtime inputs stay node-local.
STAGE=/tmp/kevinon/expo-real-17539504
export APPTAINER_CACHEDIR="$STAGE/cache" APPTAINER_TMPDIR="$STAGE/tmp"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Missing Slurm GPU visibility}"
export APPTAINERENV_JAX_PLATFORMS=cuda
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export APPTAINERENV_XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export APPTAINERENV_OPENPI_DATA_HOME=/openpi-cache
export APPTAINERENV_XDG_CACHE_HOME=/output/cache
export APPTAINERENV_HF_HOME=/output/hf
export APPTAINERENV_CUDA_CACHE_PATH=/output/cuda-cache
export APPTAINERENV_WANDB_DIR=/output/wandb
export APPTAINERENV_WANDB_MODE=online
export APPTAINERENV_WANDB_ENTITY=kevinon-stanford-university
export APPTAINERENV_WANDB_RUN_ID=znr5aqj0
export APPTAINERENV_WANDB_RESUME=allow
export APPTAINERENV_WANDB_API_KEY="$(< /iliad/u/kevinon/.config/wandb/mtexpo-api-key)"
export APPTAINERENV_WANDB_RUN_GROUP=pick-pi05base-r2-20260921
export APPTAINERENV_WANDB_JOB_TYPE=online-training
export APPTAINERENV_WANDB_TAGS=pi05_base,multi-robot,a40x2,batch8,utd20,seed42
export APPTAINERENV_OMP_NUM_THREADS=8
export APPTAINERENV_OPENBLAS_NUM_THREADS=8
export APPTAINERENV_FASTVLA_GATHER_THREADS=8
container=(apptainer exec --nv --cleanenv --containall --no-mount cwd \
 --home "$STAGE/home:/home/kevinon" --workdir "$STAGE/runtime" --pwd /opt/expo-ft \
 --bind "$STAGE/source:/opt/expo-ft:ro,$STAGE/openpi-cache:/openpi-cache,$STAGE/demo:/demo:ro,$STAGE/assets:/assets:ro,$STAGE/output:/output,$STAGE/output:/scr/kevinon/data/expo-real-17539504" "$STAGE/learner.sif")
