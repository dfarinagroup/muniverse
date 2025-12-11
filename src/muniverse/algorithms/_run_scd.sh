#!/bin/bash
set -e

# Usage:
# ./run_scd.sh docker|singularity path/to/container path/to/run_scd.py path/to/run_dir

ENGINE=$1
CONTAINER_NAME=$2
SCRIPT_PATH=$3
RUN_DIR=$4
REPO_PATH=$5
CONDA_ENV=$6

# Detect NVIDIA GPU availability on the host
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
  echo "[INFO] NVIDIA GPU detected, enabling GPU support"
  GPU_FLAG_DOCKER="--gpus all"
  GPU_FLAG_SINGULARITY="--nv"
else
  echo "[WARN] No NVIDIA GPU detected, falling back to CPU only"
  GPU_FLAG_DOCKER=""
  GPU_FLAG_SINGULARITY=""
fi

if [ "$ENGINE" == "docker" ]; then
  echo "[INFO] Running SCD with Docker"
  docker run --platform linux/amd64 --rm \
    $GPU_FLAG_DOCKER \
    -v $(realpath $SCRIPT_PATH):/opt/scd/run_scd.py \
    -v $(realpath $RUN_DIR):/run_dir/ \
    $CONTAINER_NAME \
    bash -c "source /opt/mambaforge/etc/profile.d/conda.sh && \
             conda activate decomposition && \
             cd /opt/scd/ && \
             python run_scd.py --run_dir /run_dir/"
elif [ "$ENGINE" == "singularity" ]; then
  echo "[INFO] Running SCD with Singularity"
  singularity run $GPU_FLAG_SINGULARITY --cleanenv \
    -B $(realpath $SCRIPT_PATH):/opt/scd/run_scd.py \
    -B $(realpath $RUN_DIR):/run_dir/ \
    $CONTAINER_NAME \
    bash -c "source /opt/mambaforge/etc/profile.d/conda.sh && \
             conda activate decomposition && \
             cd /opt/scd/ && \
             python run_scd.py --run_dir /run_dir/"
elif [ "$ENGINE" == "host" ]; then
  echo "[INFO] Running SCD on the host system"
  bash -c "source $(conda info --base)/etc/profile.d/conda.sh && \
           conda activate "$CONDA_ENV" && \
           cp \"$SCRIPT_PATH\" \"$REPO_PATH\" && \
           python \"$REPO_PATH/$(basename $SCRIPT_PATH)\" --run_dir \"$RUN_DIR\" "
else
  echo "ERROR: Unknown engine '$ENGINE'. Use 'docker', 'singularity' or 'host'."
  exit 1
fi
