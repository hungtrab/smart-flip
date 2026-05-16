#!/usr/bin/env bash
set -euo pipefail

# Sequential runner for cheap single-GPU instances.
# It reuses egbc_c4_b3.sh workers but runs one job at a time.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/egbc_c4_b3.sh"

GPU="${GPU:-0}"
GPU_LIST="${GPU_LIST:-$GPU,$GPU,$GPU,$GPU}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_PROJECT="${WANDB_PROJECT:-egbc_c4_b3}"
RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-./results/models}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./data/cache/hf_datasets/datasets-egbc-c4-b3}"

RUN_EGBC_BEST="${RUN_EGBC_BEST:-0}"
RUN_EGBC_SELECTED="${RUN_EGBC_SELECTED:-1}"
RUN_BC_BASELINE="${RUN_BC_BASELINE:-0}"
RUN_BC_EXTRA="${RUN_BC_EXTRA:-1}"
RUN_FLOAT_MODEL="${RUN_FLOAT_MODEL:-0}"

# Set RUN_FQ_RAW_ONCE=1 on fresh instances. Each FlatQuant worker will generate
# raw parameters only when its artifact is missing under RESULTS_MODELS_DIR.
RUN_FQ_RAW_ONCE="${RUN_FQ_RAW_ONCE:-1}"

run_one() {
  local worker="$1"
  shift

  echo "================================================================"
  echo "Starting sequential worker: $worker"
  echo "GPU=$GPU GPU_LIST=$GPU_LIST WANDB_PROJECT=$WANDB_PROJECT"
  echo "================================================================"

  env \
    GPU_LIST="$GPU_LIST" \
    PYTHON_BIN="$PYTHON_BIN" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    RESULTS_MODELS_DIR="$RESULTS_MODELS_DIR" \
    HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}" \
    WORKER="$worker" \
    "$@" \
    bash "$RUNNER"
}

flatquant_raw_dir() {
  local model_path="$1"
  local model_slug="${model_path##*/}"
  printf '%s/flatquant_raw/flatquant_raw_%s' "$RESULTS_MODELS_DIR" "$model_slug"
}

run_fq_one() {
  local worker="$1"
  local model_path="$2"
  local raw_flag=0
  local raw_dir
  raw_dir="$(flatquant_raw_dir "$model_path")"

  if [[ ! -f "$raw_dir/flat_parameters.pth" && "$RUN_FQ_RAW_ONCE" == "1" ]]; then
    raw_flag=1
  fi

  run_one "$worker" RUN_FQ_RAW="$raw_flag"
}

if [[ "$RUN_FLOAT_MODEL" == "1" ]]; then
  run_one qwen_float
  run_one llama31_float
  run_one llama3_float
fi

if [[ "$RUN_EGBC_BEST" == "1" ]]; then
  run_one qwen_awq_egbc_c4
  run_fq_one qwen_fq_egbc_c4 "Qwen/Qwen2.5-7B"
  run_one llama31_awq_egbc_c4
  run_one llama3_awq_egbc_c4
fi

if [[ "$RUN_EGBC_SELECTED" == "1" ]]; then
  run_one mistral_awq_selected_egbc_c4
  run_one llama31_awq_selected_egbc_c4
  run_one qwen_awq_selected_egbc_c4
  run_one llama3_awq_selected_egbc_c4

  run_fq_one mistral_fq_selected_egbc_c4 "mistralai/Mistral-7B-v0.3"
  run_fq_one llama31_fq_selected_egbc_c4 "meta-llama/Meta-Llama-3.1-8B"
  run_fq_one qwen_fq_selected_egbc_c4 "Qwen/Qwen2.5-7B"
  run_fq_one llama3_fq_selected_egbc_c4 "meta-llama/Meta-Llama-3-8B"
fi

if [[ "$RUN_BC_BASELINE" == "1" ]]; then
  run_one qwen_awq_bc
  run_fq_one qwen_fq_bc "Qwen/Qwen2.5-7B"
  run_one llama31_awq_bc
  run_one llama3_awq_bc
fi

if [[ "$RUN_BC_EXTRA" == "1" ]]; then
  run_one mistral_awq_bc
  run_fq_one mistral_fq_bc "mistralai/Mistral-7B-v0.3"
  run_fq_one llama31_fq_bc "meta-llama/Meta-Llama-3.1-8B"
  run_fq_one llama3_fq_bc "meta-llama/Meta-Llama-3-8B"
fi

echo "Sequential run finished."
