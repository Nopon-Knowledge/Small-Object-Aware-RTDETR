#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
JOB_NAME="${JOB_NAME:-rtdetr_wheat_tsecaf_r18}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/venvs/rtdetr_env/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-tools/train_gwhd_a40.py}"
PRESET="${PRESET:-wheat_tsecaf_r18}"
SUBMIT_BIN="${SUBMIT_BIN:-}"
QUEUE="${QUEUE:-gpu}"
GPU_COUNT="${GPU_COUNT:-1}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
EXTRA_ARGS=("$@")

mkdir -p "$LOG_DIR"

if [[ -z "$SUBMIT_BIN" ]]; then
  if command -v jsub >/dev/null 2>&1; then
    SUBMIT_BIN="jsub"
  elif command -v sbatch >/dev/null 2>&1; then
    SUBMIT_BIN="sbatch"
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

case "$SUBMIT_BIN" in
  jsub)
    jsub \
      -q "$QUEUE" \
      -gpgpu "$GPU_COUNT" \
      -J "$JOB_NAME" \
      -o "$LOG_DIR/${JOB_NAME}.%J.out" \
      -e "$LOG_DIR/${JOB_NAME}.%J.err" \
      -cwd "$PROJECT_DIR" \
      "$PYTHON_BIN" "$TRAIN_SCRIPT" --preset "$PRESET" "${EXTRA_ARGS[@]}"
    ;;
  sbatch)
    printf -v WRAP_CMD '%q ' "$PYTHON_BIN" "$TRAIN_SCRIPT" --preset "$PRESET" "${EXTRA_ARGS[@]}"
    sbatch \
      --partition="$QUEUE" \
      --gres="gpu:${GPU_COUNT}" \
      --job-name="$JOB_NAME" \
      --output="$LOG_DIR/${JOB_NAME}.%j.out" \
      --error="$LOG_DIR/${JOB_NAME}.%j.err" \
      --chdir="$PROJECT_DIR" \
      --wrap="$WRAP_CMD"
    ;;
  "")
    echo "No supported scheduler command found in PATH." >&2
    echo "Set SUBMIT_BIN=jsub or SUBMIT_BIN=sbatch after loading your cluster environment." >&2
    echo "Direct command:" >&2
    printf 'cd %q && %q %q --preset %q' "$PROJECT_DIR" "$PYTHON_BIN" "$TRAIN_SCRIPT" "$PRESET" >&2
    for arg in "${EXTRA_ARGS[@]}"; do
      printf ' %q' "$arg" >&2
    done
    printf '\n' >&2
    exit 1
    ;;
  *)
    echo "Unsupported SUBMIT_BIN: $SUBMIT_BIN" >&2
    echo "Supported values: jsub, sbatch" >&2
    exit 1
    ;;
esac
