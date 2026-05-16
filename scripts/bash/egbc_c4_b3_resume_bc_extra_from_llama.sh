#!/usr/bin/env bash
set -euo pipefail

# Resume the interrupted +BC extra phase after Mistral AWQ/FQ finished.
# Runs only:
#   - Llama3.1 FlatQuant +BC
#   - Llama3 FlatQuant +BC
# Skips a worker if its expected eval JSON already exists.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/egbc_c4_b3.sh"

GPU="${GPU:-0}"
GPU_LIST="${GPU_LIST:-$GPU,$GPU,$GPU,$GPU}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_PROJECT="${WANDB_PROJECT:-egbc_c4_b3}"
RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-./results/models}"
RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-./results/eval}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./data/cache/hf_datasets/datasets-egbc-c4-b3}"
LOG_DIR="${LOG_DIR:-./logs/egbc_c4_b3_resume}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-120}"
RUN_FQ_RAW_ONCE="${RUN_FQ_RAW_ONCE:-1}"

mkdir -p "$LOG_DIR" "$HF_DATASETS_CACHE"

flatquant_raw_dir() {
  local model_path="$1"
  local model_slug="${model_path##*/}"
  printf '%s/flatquant_raw/flatquant_raw_%s' "$RESULTS_MODELS_DIR" "$model_slug"
}

run_fq_worker_with_retry() {
  local worker="$1"
  local model_path="$2"
  local expected_json="$3"
  local raw_flag=0
  local raw_dir
  raw_dir="$(flatquant_raw_dir "$model_path")"

  if [[ -f "$expected_json" ]]; then
    echo "Skipping $worker; found $expected_json"
    return 0
  fi

  if [[ ! -f "$raw_dir/flat_parameters.pth" && "$RUN_FQ_RAW_ONCE" == "1" ]]; then
    raw_flag=1
  fi

  local attempt=1
  while (( attempt <= MAX_ATTEMPTS )); do
    echo "================================================================"
    echo "Resume worker: $worker attempt=$attempt/$MAX_ATTEMPTS"
    echo "GPU=$GPU GPU_LIST=$GPU_LIST WANDB_PROJECT=$WANDB_PROJECT RUN_FQ_RAW=$raw_flag"
    echo "expected_json=$expected_json"
    echo "raw_dir=$raw_dir"
    echo "================================================================"

    set +e
    env \
      GPU_LIST="$GPU_LIST" \
      PYTHON_BIN="$PYTHON_BIN" \
      WANDB_PROJECT="$WANDB_PROJECT" \
      RESULTS_MODELS_DIR="$RESULTS_MODELS_DIR" \
      RESULTS_EVAL_DIR="$RESULTS_EVAL_DIR" \
      HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
      HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}" \
      WORKER="$worker" \
      RUN_FQ_RAW="$raw_flag" \
      bash "$RUNNER" 2>&1 | tee "$LOG_DIR/${worker}.attempt${attempt}.log"
    status="${PIPESTATUS[0]}"
    set -e

    if [[ "$status" == "0" ]]; then
      return 0
    fi

    if [[ -f "$expected_json" ]]; then
      echo "$worker produced $expected_json despite non-zero exit; treating as complete."
      return 0
    fi

    if (( attempt == MAX_ATTEMPTS )); then
      echo "FAILED: $worker after $MAX_ATTEMPTS attempts" >&2
      return "$status"
    fi

    echo "Worker $worker failed with status $status; sleeping ${RETRY_SLEEP_SECONDS}s before retry..."
    sleep "$RETRY_SLEEP_SECONDS"
    attempt=$((attempt + 1))
    # If the first attempt generated raw params before failing, reuse them.
    if [[ -f "$raw_dir/flat_parameters.pth" ]]; then
      raw_flag=0
    fi
  done
}

run_fq_worker_with_retry \
  llama31_fq_bc \
  "meta-llama/Meta-Llama-3.1-8B" \
  "$RESULTS_EVAL_DIR/flatquant_bc_Meta-Llama-3.1-8B_b3_full.json"

run_fq_worker_with_retry \
  llama3_fq_bc \
  "meta-llama/Meta-Llama-3-8B" \
  "$RESULTS_EVAL_DIR/flatquant_bc_Meta-Llama-3-8B_b3_full.json"

echo "Resume finished."
