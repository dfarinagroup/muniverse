#!/bin/bash
set -e

# Usage:
# ./generate_recording_curated.sh path/to/run_neuromotion_curated.py path/to/input_config path/to/output_dir path/to/muaps_file.npz

SCRIPT_PATH=$1
CONFIG_PATH=$2
OUTPUT_DIR=$3
MUAPS_FILE=$4

# Check for required parameters
if [ -z "$MUAPS_FILE" ]; then
  echo "ERROR: Missing required parameters"
  echo "Usage: ./generate_recording_curated.sh path/to/run_neuromotion_curated.py path/to/input_config path/to/output_dir path/to/muaps_file.npz"
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
  echo "[INFO] NVIDIA GPU detected, enabling GPU acceleration"
  export CUDA_VISIBLE_DEVICES=0
  PYTORCH_FLAG="--pytorch-device cuda"
else
  echo "[WARN] No NVIDIA GPU detected, using CPU only"
  export CUDA_VISIBLE_DEVICES=""
  PYTORCH_FLAG="--pytorch-device cpu"
fi

# Set exponential sampling factor (optionally configurable)
EXP_FACTOR="5.0"
EXP_FACTOR_ARG="--exp-factor $EXP_FACTOR"

echo "[INFO] Running with Python directly"
echo "[INFO] Script path: $SCRIPT_PATH"
echo "[INFO] Config path: $CONFIG_PATH"
echo "[INFO] Output dir: $OUTPUT_DIR"
echo "[INFO] MUAPs file: $MUAPS_FILE"
echo "[INFO] Exponential sampling factor: $EXP_FACTOR"

# Execute the Python script
python "$(realpath $SCRIPT_PATH)" \
  "$(realpath $CONFIG_PATH)" \
  "$(realpath $OUTPUT_DIR)" \
  --muaps_file "$(realpath $MUAPS_FILE)" \
  $EXP_FACTOR_ARG \
  $PYTORCH_FLAG

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