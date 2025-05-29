#!/bin/bash
set -e

# Usage:
# ./run_scd.sh docker|singularity path/to/container path/to/run_scd.py path/to/input.npy path/to/scd_config.json path/to/output_dir

ENGINE=$1
CONTAINER_NAME=$2
SCRIPT_PATH=$3
DATA_PATH=$4
CONFIG_PATH=$5
OUTPUT_DIR=$6

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
  echo "[INFO] Running with Docker"
  docker run --platform linux/amd64 --rm \
    $GPU_FLAG_DOCKER \
    -v $(realpath $SCRIPT_PATH):/opt/scd/run_scd.py \
    -v $(realpath $DATA_PATH):/data/input.npy \
    -v $(realpath $CONFIG_PATH):/data/scd.json \
    -v $(realpath $OUTPUT_DIR):/output/ \
    $CONTAINER_NAME \
    bash -c "source /opt/mambaforge/etc/profile.d/conda.sh && \
             conda activate decomposition && \
             cd /opt/scd/ && \
             python run_scd.py /data/input.npy /data/scd.json /output/"
elif [ "$ENGINE" == "singularity" ]; then
  echo "[INFO] Running with Singularity"
  singularity run $GPU_FLAG_SINGULARITY --cleanenv \
    -B $(realpath $SCRIPT_PATH):/opt/scd/run_scd.py \
    -B $(realpath $DATA_PATH):/data/input.npy \
    -B $(realpath $CONFIG_PATH):/data/scd.json \
    -B $(realpath $OUTPUT_DIR):/output/ \
    $CONTAINER_NAME \
    bash -c "source /opt/mambaforge/etc/profile.d/conda.sh && \
             conda activate decomposition && \
             cd /opt/scd/ && \
             python run_scd.py /data/input.npy /data/scd.json /output/"
else
  echo "ERROR: Unknown engine '$ENGINE'. Use 'docker' or 'singularity'."
  exit 1
fi
