#!/usr/bin/env bash
set -euo pipefail

# Tune Llama3 FlatQuant + EGBC/Smart-Flip at 3-bit.
# Grid: 6 knee values x 5 max-flip values = 30 runs.
# This script tunes EGBC only. Bias Correction does not use these parameters.

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"

RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-./results/models}"
RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-./results/eval}"
CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-./data/cache/calibration}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./data/cache/eval}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./data/cache/hf_datasets/datasets-llama3-b3}"
LOG_DIR="${LOG_DIR:-./logs/llama3_flatquant_b3_tune_egbc}"

WANDB_PROJECT="${WANDB_PROJECT:-egbc_tune_llama3_b3}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
USE_WANDB="${USE_WANDB:-1}"

BITS="${BITS:-3}"
N_CALIB="${N_CALIB:-128}"
CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
CALIB_DATASET="${CALIB_DATASET:-c4}"
SEED="${SEED:-42}"
STRIDE="${STRIDE:-512}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
C4_SAMPLES="${C4_SAMPLES:-500}"
LM_EVAL_TASKS=(arc_challenge arc_easy boolq piqa rte)

# Default tune target matches the previous EGBC tuning runs: 5 tasks, no C4.
INCLUDE_C4="${INCLUDE_C4:-0}"
RUN_RAW="${RUN_RAW:-auto}"
RUN_FLOAT_MODEL="${RUN_FLOAT_MODEL:-0}"
SKIP_EXISTING_JSON="${SKIP_EXISTING_JSON:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-120}"
START_KNEE="${START_KNEE:-}"
START_FLIP="${START_FLIP:-}"

KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)
MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)

FLATQUANT_EPOCHS="${FLATQUANT_EPOCHS:-15}"
FLATQUANT_CALI_BSZ="${FLATQUANT_CALI_BSZ:-4}"
FLATQUANT_LR="${FLATQUANT_LR:-5e-3}"
FLATQUANT_DIAG_INIT="${FLATQUANT_DIAG_INIT:-sq_style}"
FLATQUANT_DIAG_ALPHA="${FLATQUANT_DIAG_ALPHA:-0.3}"
FLATQUANT_CALI_TRANS="${FLATQUANT_CALI_TRANS:-1}"
FLATQUANT_ADD_DIAG="${FLATQUANT_ADD_DIAG:-1}"
FLATQUANT_LWC="${FLATQUANT_LWC:-1}"
FLATQUANT_LAC="${FLATQUANT_LAC:-1}"
FLATQUANT_ORTHOGONAL_MAP="${FLATQUANT_ORTHOGONAL_MAP:-matrix_exp}"
FLATQUANT_USE_TRIVIALIZATION="${FLATQUANT_USE_TRIVIALIZATION:-1}"

export FLATQUANT_ORTHOGONAL_MAP
export FLATQUANT_USE_TRIVIALIZATION
export HF_DATASETS_CACHE
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

mkdir -p "$RESULTS_MODELS_DIR" "$RESULTS_EVAL_DIR" "$CALIBRATION_CACHE_DIR" "$EVAL_CACHE_DIR" "$HF_DATASETS_CACHE" "$LOG_DIR"

MODEL_SLUG="${MODEL_PATH##*/}"
RAW_RUN_NAME="${RAW_RUN_NAME:-flatquant_raw_${MODEL_SLUG}_b${BITS}}"
RAW_MODEL_DIR="${RAW_MODEL_DIR:-${RESULTS_MODELS_DIR}/flatquant_raw/${RAW_RUN_NAME}}"

wandb_args=()
if [[ "$USE_WANDB" == "1" ]]; then
  wandb_args+=(--use-wandb --wandb-project "$WANDB_PROJECT")
  if [[ -n "$WANDB_ENTITY" ]]; then
    wandb_args+=(--wandb-entity "$WANDB_ENTITY")
  fi
else
  wandb_args+=(--no-wandb)
fi

add_flatquant_args() {
  local -n args_ref=$1
  args_ref+=(
    --flatquant-epochs "$FLATQUANT_EPOCHS"
    --flatquant-cali-bsz "$FLATQUANT_CALI_BSZ"
    --flatquant-lr "$FLATQUANT_LR"
    --flatquant-diag-init "$FLATQUANT_DIAG_INIT"
    --flatquant-diag-alpha "$FLATQUANT_DIAG_ALPHA"
  )
  [[ "$FLATQUANT_CALI_TRANS" == "1" ]] && args_ref+=(--flatquant-cali-trans) || args_ref+=(--no-flatquant-cali-trans)
  [[ "$FLATQUANT_ADD_DIAG" == "1" ]] && args_ref+=(--flatquant-add-diag) || args_ref+=(--no-flatquant-add-diag)
  [[ "$FLATQUANT_LWC" == "1" ]] && args_ref+=(--flatquant-lwc) || args_ref+=(--no-flatquant-lwc)
  [[ "$FLATQUANT_LAC" == "1" ]] && args_ref+=(--flatquant-lac) || args_ref+=(--no-flatquant-lac)
}

common_quant_args=(
  --model-path "$MODEL_PATH"
  --models-root "$MODELS_ROOT"
  --results-models-dir "$RESULTS_MODELS_DIR"
  --results-eval-dir "$RESULTS_EVAL_DIR"
  --calibration-cache-dir "$CALIBRATION_CACHE_DIR"
  --eval-cache-dir "$EVAL_CACHE_DIR"
  --calib-dataset "$CALIB_DATASET"
  --n-calib "$N_CALIB"
  --calib-seqlen "$CALIB_SEQLEN"
  --seed "$SEED"
  --stride "$STRIDE"
  --max-length "$MAX_LENGTH"
  --c4-samples "$C4_SAMPLES"
  --bits "$BITS"
  --lm-eval-tasks "${LM_EVAL_TASKS[@]}"
  "${wandb_args[@]}"
)

if [[ "$INCLUDE_C4" == "1" ]]; then
  common_quant_args+=(--include-c4)
else
  common_quant_args+=(--no-c4)
fi

run_raw=false
if [[ "$RUN_RAW" == "1" ]]; then
  run_raw=true
elif [[ "$RUN_RAW" == "auto" && ! -f "$RAW_MODEL_DIR/flat_parameters.pth" ]]; then
  run_raw=true
fi

echo "================================================================"
echo "Llama3 FlatQuant + EGBC tune b${BITS}"
echo "model=$MODEL_PATH gpu=$GPU raw=$RUN_RAW include_c4=$INCLUDE_C4"
echo "raw_dir=$RAW_MODEL_DIR"
echo "wandb_project=$WANDB_PROJECT"
echo "grid: ${#KNEE_VALUES[@]} x ${#MAX_FLIP_VALUES[@]} = $((${#KNEE_VALUES[@]} * ${#MAX_FLIP_VALUES[@]})) runs"
if [[ -n "$START_KNEE" || -n "$START_FLIP" ]]; then
  echo "resume_start: knee=${START_KNEE:-<unset>} flip=${START_FLIP:-<unset>}"
fi
echo "================================================================"

if [[ "$RUN_FLOAT_MODEL" == "1" ]]; then
  float_args=(
    --model-path "$MODEL_PATH"
    --models-root "$MODELS_ROOT"
    --results-eval-dir "$RESULTS_EVAL_DIR"
    --eval-cache-dir "$EVAL_CACHE_DIR"
    --seed "$SEED"
    --stride "$STRIDE"
    --max-length "$MAX_LENGTH"
    --c4-samples "$C4_SAMPLES"
    --lm-eval-tasks "${LM_EVAL_TASKS[@]}"
    "${wandb_args[@]}"
    --run-name "llama3_float_${MODEL_SLUG}_b${BITS}_tune_ref"
  )
  [[ "$INCLUDE_C4" == "1" ]] && float_args+=(--include-c4) || float_args+=(--no-c4)
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" main.py float_model "${float_args[@]}" 2>&1 | tee "$LOG_DIR/float_${MODEL_SLUG}.log"
fi

if [[ "$run_raw" == "true" ]]; then
  raw_args=(
    "${common_quant_args[@]}"
    --origin-method flatquant
    --post-correction none
    --run-name "$RAW_RUN_NAME"
    --no-lm-eval
  )
  add_flatquant_args raw_args
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" main.py quantize "${raw_args[@]}" 2>&1 | tee "$LOG_DIR/${RAW_RUN_NAME}.log"
elif [[ ! -f "$RAW_MODEL_DIR/flat_parameters.pth" ]]; then
  echo "Missing FlatQuant raw artifact: $RAW_MODEL_DIR/flat_parameters.pth" >&2
  echo "Use RUN_RAW=1/RUN_RAW=auto to generate it or set RAW_MODEL_DIR." >&2
  exit 1
else
  echo "Skipping raw; found $RAW_MODEL_DIR/flat_parameters.pth"
fi

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

resume_gate_open=1
if [[ -n "$START_KNEE" || -n "$START_FLIP" ]]; then
  resume_gate_open=0
fi

for knee in "${KNEE_VALUES[@]}"; do
  for flip in "${MAX_FLIP_VALUES[@]}"; do
    if [[ "$resume_gate_open" == "0" ]]; then
      if [[ "$knee" == "$START_KNEE" && "$flip" == "$START_FLIP" ]]; then
        resume_gate_open=1
      else
        echo "Skipping before resume point: k=$knee f=$flip"
        continue
      fi
    fi

    run_name="llama3_fq_egbc_${MODEL_SLUG}_b${BITS}_k${knee}_f${flip}"
    egbc_args=(
      "${common_quant_args[@]}"
      --origin-method flatquant
      --post-correction smart_flip
      --flatquant-raw-path "$RAW_MODEL_DIR"
      --knee-tolerance "$knee"
      --max-flip-percent "$flip"
      --run-name "$run_name"
    )
    add_flatquant_args egbc_args
    run_with_retry "$run_name" "${egbc_args[@]}"
  done
done

echo "Tune finished. Logs: $LOG_DIR"
