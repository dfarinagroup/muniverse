#!/bin/bash
set -e

# Usage:
# ./generate_recording_new.sh docker|singularity pranavm19/muniverse:neuromotion path/to/run_neuromotion_new.py path/to/run_dir [cache_dir]

ENGINE=$1
CONTAINER_NAME=$2
SCRIPT_PATH=$3
RUN_DIR=$4

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
  docker run -it --platform linux/amd64 --rm \
    $GPU_FLAG_DOCKER \
    -v $(realpath $SCRIPT_PATH):/opt/NeuroMotion/run_neuromotion_new.py \
    -v $(realpath $RUN_DIR):/run_dir/ \
    $CONTAINER_NAME \
    bash -c "source /opt/mambaforge/etc/profile.d/conda.sh && \
             conda activate NeuroMotion && \
             cd /opt/NeuroMotion/ && \
             python run_neuromotion_new.py --run_dir /run_dir/"
elif [ "$ENGINE" == "singularity" ]; then
  echo "[INFO] Running with Singularity"
  singularity run $GPU_FLAG_SINGULARITY --cleanenv \
    -B $(realpath $SCRIPT_PATH):/opt/NeuroMotion/run_neuromotion_new.py \
    -B $(realpath $RUN_DIR):/run_dir/ \
    $CONTAINER_NAME \
    bash -c "source /opt/mambaforge/etc/profile.d/conda.sh && \
             conda activate NeuroMotion && \
             cd /opt/NeuroMotion/ && \
             python run_neuromotion_new.py --run_dir /run_dir/"
else
  echo "ERROR: Unknown engine '$ENGINE'. Use 'docker' or 'singularity'."
  exit 1
fi

# Check if output was generated successfully
if [ -f "$RUN_DIR/emg_data.npz" ]; then
  echo "[INFO] EMG data generated successfully at $RUN_DIR/emg_data.npz"
else
  echo "[ERROR] Container did not produce EMG data"
  exit 1
fi
