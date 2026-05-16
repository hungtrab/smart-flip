#!/usr/bin/env bash
set -euo pipefail

# Overnight runner for final weight-only RTN vs RTN+SmartFlip.
# Runs W3A16 first, then W4A16. Each child script skips completed JSON files
# and removes heavy model weights after evaluation by default.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-./results/models}"
RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-./results/eval}"
CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-./data/cache/calibration}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./data/cache/eval}"

WANDB_PROJECT_B3="${WANDB_PROJECT_B3:-rtn_flip_w3a16}"
WANDB_PROJECT_B4="${WANDB_PROJECT_B4:-rtn_flip_w4a16}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
USE_WANDB="${USE_WANDB:-1}"

RUN_RAW="${RUN_RAW:-auto}"
RUN_FLIP="${RUN_FLIP:-1}"
SKIP_EXISTING_JSON="${SKIP_EXISTING_JSON:-1}"
CLEAN_MODEL_ARTIFACTS="${CLEAN_MODEL_ARTIFACTS:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-120}"
KNEE="${KNEE:-0.0}"
MAX_FLIP="${MAX_FLIP:-0.05}"

run_bits() {
  local bits="$1"
  local project="$2"
  local script="$SCRIPT_DIR/rtn_flip_b${bits}.sh"

  echo "================================================================"
  echo "Starting RTN W${bits}A16 sequential block"
  echo "script=$script"
  echo "wandb_project=$project"
  echo "gpu=$GPU"
  echo "================================================================"

  env \
    PYTHON_BIN="$PYTHON_BIN" \
    GPU="$GPU" \
    MODELS_ROOT="$MODELS_ROOT" \
    RESULTS_MODELS_DIR="$RESULTS_MODELS_DIR" \
    RESULTS_EVAL_DIR="$RESULTS_EVAL_DIR" \
    CALIBRATION_CACHE_DIR="$CALIBRATION_CACHE_DIR" \
    EVAL_CACHE_DIR="$EVAL_CACHE_DIR" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./data/cache/hf_datasets/datasets-rtn-flip-w${bits}a16}" \
    PRESERVE_HF_DATASETS_CACHE="${PRESERVE_HF_DATASETS_CACHE:-1}" \
    LOG_DIR="${LOG_DIR:-./logs/rtn_flip_w${bits}a16}" \
    WANDB_PROJECT="$project" \
    WANDB_ENTITY="$WANDB_ENTITY" \
    USE_WANDB="$USE_WANDB" \
    BITS="$bits" \
    FLATQUANT_A_BITS=16 \
    RUN_RAW="$RUN_RAW" \
    RUN_FLIP="$RUN_FLIP" \
    SKIP_EXISTING_JSON="$SKIP_EXISTING_JSON" \
    CLEAN_MODEL_ARTIFACTS="$CLEAN_MODEL_ARTIFACTS" \
    KEEP_MODEL_ARTIFACTS_FOR="${KEEP_MODEL_ARTIFACTS_FOR:-}" \
    MAX_ATTEMPTS="$MAX_ATTEMPTS" \
    RETRY_SLEEP_SECONDS="$RETRY_SLEEP_SECONDS" \
    KNEE="$KNEE" \
    MAX_FLIP="$MAX_FLIP" \
    MODEL_PATH="${MODEL_PATH:-}" \
    RUN_LLAMA3="${RUN_LLAMA3:-0}" \
    bash "$script"
}

run_bits 3 "$WANDB_PROJECT_B3"
run_bits 4 "$WANDB_PROJECT_B4"

echo "Finished RTN W3A16 + W4A16 sequential run."
