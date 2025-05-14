#!/bin/bash
set -e

# Usage:
# ./generate_records_hybrid.sh docker|singularity container_name path/to/_run_hybrid.py path/to/input_config path/to/output_dir path/to/muaps_file.npz

ENGINE=$1
CONTAINER_NAME=$2
SCRIPT_PATH=$3
CONFIG_PATH=$4
OUTPUT_DIR=$5
MUAPS_FILE=$6

# Check for required parameters
if [ -z "$MUAPS_FILE" ]; then
  echo "ERROR: Missing required parameters"
  echo "Usage: ./generate_records_hybrid.sh docker|singularity container_name path/to/_run_hybrid.py path/to/input_config path/to/output_dir path/to/muaps_file.npz"
  exit 1
fi

# Verify files exist
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "ERROR: Script file not found: $SCRIPT_PATH"
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "ERROR: Config file not found: $CONFIG_PATH"
  exit 1
fi

if [ ! -f "$MUAPS_FILE" ]; then
  echo "ERROR: MUAPs file not found: $MUAPS_FILE"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Detect NVIDIA GPU availability on the host
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
  echo "[INFO] NVIDIA GPU detected, enabling GPU support"
  GPU_FLAG_DOCKER="--gpus all"
  GPU_FLAG_SINGULARITY="--nv"
  PYTORCH_FLAG="--pytorch-device cuda"
else
  echo "[WARN] No NVIDIA GPU detected, falling back to CPU only"
  GPU_FLAG_DOCKER=""
  GPU_FLAG_SINGULARITY=""
  PYTORCH_FLAG="--pytorch-device cpu"
fi

# Set exponential sampling factor
EXP_FACTOR="5.0"
EXP_FACTOR_ARG="--exp-factor $EXP_FACTOR"

# Get directory of the muaps file for mounting
MUAPS_DIR=$(dirname "$(realpath $MUAPS_FILE)")
MUAPS_FILENAME=$(basename "$(realpath $MUAPS_FILE)")

if [ "$ENGINE" == "docker" ]; then
  echo "[INFO] Running with Docker"
  docker run --platform linux/amd64 --rm \
    $GPU_FLAG_DOCKER \
    -v $(realpath $SCRIPT_PATH):/opt/NeuroMotion/run_script.py \
    -v $(realpath $CONFIG_PATH):/data/input_config.json \
    -v $(realpath $OUTPUT_DIR):/output/ \
    -v $MUAPS_DIR:/muaps/ \
    $CONTAINER_NAME \
    bash -c "source /opt/mambaforge/etc/profile.d/conda.sh && \
             conda activate NeuroMotion && \
             cd /opt/NeuroMotion/ && \
             python run_script.py /data/input_config.json /output/ \
             --muaps_file /muaps/$MUAPS_FILENAME \
             $EXP_FACTOR_ARG \
             $PYTORCH_FLAG"
elif [ "$ENGINE" == "singularity" ]; then
  echo "[INFO] Running with Singularity"
  singularity run $GPU_FLAG_SINGULARITY --cleanenv \
    -B $(realpath $SCRIPT_PATH):/opt/NeuroMotion/run_script.py \
    -B $(realpath $CONFIG_PATH):/data/input_config.json \
    -B $(realpath $OUTPUT_DIR):/output/ \
    -B $MUAPS_DIR:/muaps/ \
    $CONTAINER_NAME \
    bash -c "source /opt/mambaforge/etc/profile.d/conda.sh && \
             conda activate NeuroMotion && \
             cd /opt/NeuroMotion/ && \
             python run_script.py /data/input_config.json /output/ \
             --muaps_file /muaps/$MUAPS_FILENAME \
             $EXP_FACTOR_ARG \
             $PYTORCH_FLAG"
else
  echo "[INFO] Running with Python directly (no container)"
  echo "[INFO] Script path: $SCRIPT_PATH"
  echo "[INFO] Config path: $CONFIG_PATH"
  echo "[INFO] Output dir: $OUTPUT_DIR"
  echo "[INFO] MUAPs file: $MUAPS_FILE"
  echo "[INFO] Exponential sampling factor: $EXP_FACTOR"

  # Execute the Python script directly
  python "$(realpath $SCRIPT_PATH)" \
    "$(realpath $CONFIG_PATH)" \
    "$(realpath $OUTPUT_DIR)" \
    --muaps_file "$(realpath $MUAPS_FILE)" \
    $EXP_FACTOR_ARG \
    $PYTORCH_FLAG
fi

# Check if the script executed successfully
if [ $? -eq 0 ]; then
  echo "[INFO] Data generation completed successfully"
else
  echo "[ERROR] Data generation failed"
  exit 1
fi

# Find and report the output directory
LATEST_RUN=$(find "$(realpath $OUTPUT_DIR)" -maxdepth 1 -type d -name "run_*" | sort -r | head -n 1)
if [ -n "$LATEST_RUN" ]; then
  echo "[INFO] Output available at: $LATEST_RUN"
  
  # Find and report any metadata files
  METADATA_FILES=$(find "$LATEST_RUN" -name "*_metadata.json")
  if [ -n "$METADATA_FILES" ]; then
    echo "[INFO] Metadata files found:"
    for file in $METADATA_FILES; do
      echo "        $file"
    done
  fi
else
  echo "[WARN] No output directory found"
fi