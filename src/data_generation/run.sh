#!/bin/bash
set -e

# Usage:
# ./run.sh docker|singularity path/to/run_neuromotion.py path/to/input_config path/to/output_dir

ENGINE=$1
CONTAINER_NAME=$2
SCRIPT_PATH=$3
CONFIG_PATH=$4
OUTPUT_DIR=$5

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
    -v $(realpath $SCRIPT_PATH):/opt/NeuroMotion/run_neuromotion.py \
    -v $(realpath $CONFIG_PATH):/data/input_config.json \
    -v $(realpath $OUTPUT_DIR):/output/ \
    $CONTAINER_NAME \
    bash -c "source /opt/mambaforge/etc/profile.d/conda.sh && \
             conda activate NeuroMotion && \
             cd /opt/NeuroMotion/ && \
             python run_neuromotion.py /data/input_config.json /output/"
elif [ "$ENGINE" == "singularity" ]; then
  echo "[INFO] Running with Singularity"
  singularity run $GPU_FLAG_SINGULARITY --cleanenv \
    -B $(realpath $SCRIPT_PATH):/opt/NeuroMotion/run_neuromotion.py \
    -B $(realpath $CONFIG_PATH):/data/input_config.json \
    -B $(realpath $OUTPUT_DIR):/output/ \
    $CONTAINER_NAME \
    bash -c "source /opt/mambaforge/etc/profile.d/conda.sh && \
             conda activate NeuroMotion && \
             cd /opt/NeuroMotion/ && \
             python run_neuromotion.py /data/input_config.json /output/"
else
  echo "ERROR: Unknown engine '$ENGINE'. Use 'docker' or 'singularity'."
  exit 1
fi
