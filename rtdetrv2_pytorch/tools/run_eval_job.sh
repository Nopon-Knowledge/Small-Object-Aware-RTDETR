#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <config> <checkpoint> <output_dir> [device]" >&2
  exit 2
fi

CONFIG_PATH="$1"
CHECKPOINT_PATH="$2"
OUTPUT_DIR="$3"
DEVICE="${4:-cuda:0}"
PYTHON_BIN="${RTDETR_PYTHON:-$HOME/venvs/rtdetr_env/bin/python}"

"$PYTHON_BIN" tools/test_gwhd_a40.py \
  -c "$CONFIG_PATH" \
  -r "$CHECKPOINT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  -d "$DEVICE"
