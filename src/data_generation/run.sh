#!/bin/bash
set -e

# Usage:
# ./run.sh docker|singularity path/to/run_neuromotion.py path/to/input_config path/to/output_dir

ENGINE=$1
CONTAINER_NAME=$2
SCRIPT_PATH=$3
CONFIG_PATH=$4
OUTPUT_DIR=$5

if [ "$ENGINE" == "docker" ]; then
  echo "[INFO] Running with Docker"
  docker run --platform linux/amd64 --rm -it \
    --gpus all \
    -v $(realpath $SCRIPT_PATH):/opt/NeuroMotion/run_neuromotion.py \
    -v $(realpath $CONFIG_PATH):/data/input_config.yml \
    -v $(realpath $OUTPUT_DIR):/output/ \
    $CONTAINER_NAME \
    bash -c "source /opt/mambaforge/etc/profile.d/conda.sh && \
             conda activate NeuroMotion && \
             cd /opt/NeuroMotion/ && \
             python run_neuromotion.py /data/input_config.yml /output/"
elif [ "$ENGINE" == "singularity" ]; then
  echo "[INFO] Running with Singularity"
  singularity run --nv --cleanenv \
    -B $(realpath $SCRIPT_PATH):/opt/NeuroMotion/run_neuromotion.py \
    -B $(realpath $CONFIG_PATH):/data/input_config.yml \
    -B $(realpath $OUTPUT_DIR):/output/ \
    $CONTAINER_NAME.sif \
    bash -c "source /opt/mambaforge/etc/profile.d/conda.sh && \
             conda activate NeuroMotion && \
             cd /opt/NeuroMotion/ && \
             python run_neuromotion.py /data/input_config.yml /output/"
else
  echo "ERROR: Unknown engine '$ENGINE'. Use 'docker' or 'singularity'."
  exit 1
fi
