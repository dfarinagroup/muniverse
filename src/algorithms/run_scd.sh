#!/bin/bash
set -e

# Usage:
# ./run_scd.sh path/to/run_scd.py path/to/input.npy path/to/output_dir

SCRIPT_PATH=$1
DATA_PATH=$2
OUTPUT_DIR=$3

if [ -z "$SCRIPT_PATH" ] || [ -z "$DATA_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: ./run_scd.sh run_scd.py input.npy output_dir"
  exit 1
fi

docker run -it \
  -v $(realpath $SCRIPT_PATH):/opt/scd/run_scd.py \
  -v $(realpath $DATA_PATH):/data/input.npy \
  -v $(realpath $OUTPUT_DIR):/output/ \
  pranavm19/muniverse-test:scd \
  bash -c "source /opt/mambaforge/etc/profile.d/conda.sh && \
           conda activate decomposition && \
           cd /opt/scd/ && \
           python run_scd.py /data/input.npy /output/"
