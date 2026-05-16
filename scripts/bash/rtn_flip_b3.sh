#!/usr/bin/env bash
set -euo pipefail

# Final full evaluation: naive RTN vs RTN + SmartFlip at 3-bit.
#
# Implementation note:
# The repo does not expose a separate `origin-method rtn`. We use the
# FlatQuant RTN backend with all FlatQuant transform learning disabled, so the
# raw artifact is plain RTN and the +flip run applies SmartFlip on that same RTN
# state. This script runs full metrics: WikiText-2 PPL, C4 PPL, and the
# five lm-eval tasks. Llama3 is intentionally excluded from the default model
# list because it is being handled in the isolated for_llama3 repo/session.

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-./results/models}"
RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-./results/eval}"
CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-./data/cache/calibration}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./data/cache/eval}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./data/cache/hf_datasets/datasets-rtn-flip-b3}"
LOG_DIR="${LOG_DIR:-./logs/rtn_flip_b3}"

WANDB_PROJECT="${WANDB_PROJECT:-rtn_flip_b3}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
USE_WANDB="${USE_WANDB:-1}"

BITS="${BITS:-3}"
FLATQUANT_A_BITS="${FLATQUANT_A_BITS:-16}"
ACT_SUFFIX="a${FLATQUANT_A_BITS}"
N_CALIB="${N_CALIB:-128}"
CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
CALIB_DATASET="${CALIB_DATASET:-c4}"
SEED="${SEED:-42}"
STRIDE="${STRIDE:-512}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
C4_SAMPLES="${C4_SAMPLES:-500}"
LM_EVAL_TASKS=(arc_challenge arc_easy boolq piqa rte)

# Fixed SmartFlip config selected after PPL-first verification.
KNEE="${KNEE:-0.0}"
MAX_FLIP="${MAX_FLIP:-0.05}"

RUN_RAW="${RUN_RAW:-auto}"
RUN_FLIP="${RUN_FLIP:-1}"
SKIP_EXISTING_JSON="${SKIP_EXISTING_JSON:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-120}"
RUN_LLAMA3="${RUN_LLAMA3:-0}"
CLEAN_MODEL_ARTIFACTS="${CLEAN_MODEL_ARTIFACTS:-1}"
KEEP_MODEL_ARTIFACTS_FOR="${KEEP_MODEL_ARTIFACTS_FOR:-}"

export HF_DATASETS_CACHE
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

mkdir -p "$RESULTS_MODELS_DIR" "$RESULTS_EVAL_DIR" "$CALIBRATION_CACHE_DIR" "$EVAL_CACHE_DIR" "$HF_DATASETS_CACHE" "$LOG_DIR"

MODELS=(
  "mistralai/Mistral-7B-v0.3"
  "meta-llama/Meta-Llama-3.1-8B"
  "Qwen/Qwen2.5-7B"
)

if [[ -n "${MODEL_PATH:-}" ]]; then
  MODELS=("$MODEL_PATH")
elif [[ "$RUN_LLAMA3" == "1" ]]; then
  MODELS+=("meta-llama/Meta-Llama-3-8B")
fi

wandb_args=()
if [[ "$USE_WANDB" == "1" ]]; then
  wandb_args+=(--use-wandb --wandb-project "$WANDB_PROJECT")
  if [[ -n "$WANDB_ENTITY" ]]; then
    wandb_args+=(--wandb-entity "$WANDB_ENTITY")
  fi
else
  wandb_args+=(--no-wandb)
fi

common_args() {
  local model_path="$1"
  printf '%s\0' \
    --model-path "$model_path" \
    --models-root "$MODELS_ROOT" \
    --results-models-dir "$RESULTS_MODELS_DIR" \
    --results-eval-dir "$RESULTS_EVAL_DIR" \
    --calibration-cache-dir "$CALIBRATION_CACHE_DIR" \
    --eval-cache-dir "$EVAL_CACHE_DIR" \
    --calib-dataset "$CALIB_DATASET" \
    --n-calib "$N_CALIB" \
    --calib-seqlen "$CALIB_SEQLEN" \
    --seed "$SEED" \
    --stride "$STRIDE" \
    --max-length "$MAX_LENGTH" \
    --c4-samples "$C4_SAMPLES" \
    --bits "$BITS" \
    --flatquant-a-bits "$FLATQUANT_A_BITS" \
    --include-c4 \
    --include-lm-eval \
    --lm-eval-tasks "${LM_EVAL_TASKS[@]}" \
    "${wandb_args[@]}"
}

read_common_args() {
  local model_path="$1"
  local -n out_ref=$2
  mapfile -d '' -t out_ref < <(common_args "$model_path")
}

add_naive_rtn_args() {
  local -n args_ref=$1
  args_ref+=(
    --no-flatquant-cali-trans
    --no-flatquant-add-diag
    --no-flatquant-lwc
    --no-flatquant-lac
    --flatquant-diag-init one_style
  )
}

run_with_retry() {
  local run_name="$1"
  shift
  local expected_json="$RESULTS_EVAL_DIR/${run_name}.json"

  if [[ "$SKIP_EXISTING_JSON" == "1" && -f "$expected_json" ]]; then
    echo "Skipping existing run: $run_name"
    return 0
  fi

  local attempt=1
  while (( attempt <= MAX_ATTEMPTS )); do
    echo "==> $run_name attempt=$attempt/$MAX_ATTEMPTS"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" main.py quantize "$@" 2>&1 | tee "$LOG_DIR/${run_name}.attempt${attempt}.log"
    local status="${PIPESTATUS[0]}"
    set -e

    if [[ "$status" == "0" || -f "$expected_json" ]]; then
      return 0
    fi
    if (( attempt == MAX_ATTEMPTS )); then
      echo "FAILED: $run_name after $MAX_ATTEMPTS attempts" >&2
      return "$status"
    fi
    echo "Retrying $run_name after ${RETRY_SLEEP_SECONDS}s..."
    sleep "$RETRY_SLEEP_SECONDS"
    attempt=$((attempt + 1))
  done
}

cleanup_model_artifacts() {
  local variant="$1"
  local run_name="$2"
  local model_dir="$RESULTS_MODELS_DIR/$variant/$run_name"

  if [[ "$CLEAN_MODEL_ARTIFACTS" != "1" || ! -d "$model_dir" ]]; then
    return 0
  fi

  if [[ -n "$KEEP_MODEL_ARTIFACTS_FOR" ]]; then
    local keep_pattern
    IFS=',' read -ra keep_patterns <<< "$KEEP_MODEL_ARTIFACTS_FOR"
    for keep_pattern in "${keep_patterns[@]}"; do
      if [[ -n "$keep_pattern" && "$run_name" == *"$keep_pattern"* ]]; then
        echo "Keeping model artifacts for $run_name because it matches KEEP_MODEL_ARTIFACTS_FOR=$KEEP_MODEL_ARTIFACTS_FOR"
        return 0
      fi
    done
  fi

  echo "Cleaning heavy model artifacts under: $model_dir"
  find "$model_dir" -type f \( \
    -name "pytorch_model*.bin" -o \
    -name "*.safetensors" -o \
    -name "*.pth" -o \
    -name "*.pt" -o \
    -name "*.ckpt" \
  \) -print -delete
}

run_one_model() {
  local model_path="$1"
  local slug="${model_path##*/}"
  local raw_run="rtn_raw_${slug}_b${BITS}_${ACT_SUFFIX}"
  local raw_dir="$RESULTS_MODELS_DIR/flatquant_raw/$raw_run"
  local raw_json="$RESULTS_EVAL_DIR/${raw_run}.json"
  local args=()

  echo "================================================================"
  echo "RTN vs RTN+Flip b${BITS}: $model_path"
  echo "flatquant_a_bits=$FLATQUANT_A_BITS act_suffix=$ACT_SUFFIX"
  echo "raw_dir=$raw_dir"
  echo "wandb_project=$WANDB_PROJECT"
  echo "================================================================"

  local run_raw=false
  if [[ "$RUN_RAW" == "1" ]]; then
    run_raw=true
  elif [[ "$RUN_RAW" == "auto" && ! -f "$raw_json" ]]; then
    run_raw=true
  fi

  if [[ "$run_raw" == "true" ]]; then
    args=()
    read_common_args "$model_path" args
    add_naive_rtn_args args
    run_with_retry "$raw_run" \
      "${args[@]}" \
      --origin-method flatquant \
      --post-correction none \
      --run-name "$raw_run"
  elif [[ ! -f "$raw_json" ]]; then
    echo "Missing RTN raw eval JSON: $raw_json" >&2
    echo "Use RUN_RAW=1/RUN_RAW=auto, or set RESULTS_EVAL_DIR to where the raw result exists." >&2
    return 1
  else
    echo "Skipping raw; found $raw_json"
  fi
  cleanup_model_artifacts "flatquant_raw" "$raw_run"

  if [[ "$RUN_FLIP" != "1" ]]; then
    return 0
  fi

  local flip_run="rtn_flip_${slug}_b${BITS}_${ACT_SUFFIX}_k${KNEE}_f${MAX_FLIP}"
  args=()
  read_common_args "$model_path" args
  add_naive_rtn_args args
  echo "Running fixed RTN+Flip config: knee=$KNEE max_flip=$MAX_FLIP"
  run_with_retry "$flip_run" \
    "${args[@]}" \
    --origin-method flatquant \
    --post-correction smart_flip \
    --knee-tolerance "$KNEE" \
    --max-flip-percent "$MAX_FLIP" \
    --run-name "$flip_run"
  cleanup_model_artifacts "flatquant_smart_flip" "$flip_run"
}

for model in "${MODELS[@]}"; do
  run_one_model "$model"
done

echo "Done. Logs: $LOG_DIR"
