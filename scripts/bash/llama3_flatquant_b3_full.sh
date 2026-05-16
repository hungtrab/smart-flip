#!/usr/bin/env bash
set -euo pipefail

# Llama3 FlatQuant 3-bit pipeline:
#   1) FlatQuant raw artifact
#   2) FlatQuant + EGBC/Smart-Flip with tuned k/f
#   3) FlatQuant + Bias Correction

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"

RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-./results/models}"
RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-./results/eval}"
CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-./data/cache/calibration}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./data/cache/eval}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./data/cache/hf_datasets/datasets-llama3-b3}"
LOG_DIR="${LOG_DIR:-./logs/llama3_flatquant_b3}"

WANDB_PROJECT="${WANDB_PROJECT:-egbc_c4_b3}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
USE_WANDB="${USE_WANDB:-1}"

BITS="${BITS:-3}"
KNEE="${KNEE:-0.02}"
MAX_FLIP="${MAX_FLIP:-0.05}"
BIAS_CORRECTION_SAMPLES="${BIAS_CORRECTION_SAMPLES:-4096}"

N_CALIB="${N_CALIB:-128}"
CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
CALIB_DATASET="${CALIB_DATASET:-c4}"
SEED="${SEED:-42}"
STRIDE="${STRIDE:-512}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
C4_SAMPLES="${C4_SAMPLES:-500}"
LM_EVAL_TASKS=(arc_challenge arc_easy boolq piqa rte)

RUN_RAW="${RUN_RAW:-auto}"
RUN_EGBC="${RUN_EGBC:-1}"
RUN_BC="${RUN_BC:-1}"

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
EGBC_RUN_NAME="${EGBC_RUN_NAME:-llama3_fq_egbc_${MODEL_SLUG}_b${BITS}_k${KNEE}_f${MAX_FLIP}}"
BC_RUN_NAME="${BC_RUN_NAME:-llama3_fq_bc_${MODEL_SLUG}_b${BITS}_full}"

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
  "${wandb_args[@]}"
)

run_raw=false
if [[ "$RUN_RAW" == "1" ]]; then
  run_raw=true
elif [[ "$RUN_RAW" == "auto" && ! -f "$RAW_MODEL_DIR/flat_parameters.pth" ]]; then
  run_raw=true
fi

echo "================================================================"
echo "Llama3 FlatQuant b${BITS} pipeline"
echo "model=$MODEL_PATH gpu=$GPU raw=$RUN_RAW egbc=$RUN_EGBC bc=$RUN_BC"
echo "raw_dir=$RAW_MODEL_DIR"
echo "egbc: k=$KNEE f=$MAX_FLIP"
echo "wandb_project=$WANDB_PROJECT"
echo "================================================================"

if [[ "$run_raw" == "true" ]]; then
  raw_args=(
    "${common_quant_args[@]}"
    --origin-method flatquant
    --post-correction none
    --run-name "$RAW_RUN_NAME"
    --no-lm-eval
    --no-c4
  )
  add_flatquant_args raw_args
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" main.py quantize "${raw_args[@]}" 2>&1 | tee "$LOG_DIR/${RAW_RUN_NAME}.log"
else
  if [[ ! -f "$RAW_MODEL_DIR/flat_parameters.pth" ]]; then
    echo "Missing FlatQuant raw artifact: $RAW_MODEL_DIR/flat_parameters.pth" >&2
    echo "Use RUN_RAW=1 or RUN_RAW=auto to generate it." >&2
    exit 1
  fi
  echo "Skipping raw; found $RAW_MODEL_DIR/flat_parameters.pth"
fi

if [[ "$RUN_EGBC" == "1" ]]; then
  egbc_args=(
    "${common_quant_args[@]}"
    --origin-method flatquant
    --post-correction smart_flip
    --flatquant-raw-path "$RAW_MODEL_DIR"
    --knee-tolerance "$KNEE"
    --max-flip-percent "$MAX_FLIP"
    --lm-eval-tasks "${LM_EVAL_TASKS[@]}"
    --run-name "$EGBC_RUN_NAME"
  )
  add_flatquant_args egbc_args
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" main.py quantize "${egbc_args[@]}" 2>&1 | tee "$LOG_DIR/${EGBC_RUN_NAME}.log"
fi

if [[ "$RUN_BC" == "1" ]]; then
  bc_args=(
    "${common_quant_args[@]}"
    --origin-method flatquant
    --post-correction bias_correction
    --flatquant-raw-path "$RAW_MODEL_DIR"
    --bias-correction-samples "$BIAS_CORRECTION_SAMPLES"
    --lm-eval-tasks "${LM_EVAL_TASKS[@]}"
    --run-name "$BC_RUN_NAME"
  )
  add_flatquant_args bc_args
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" main.py quantize "${bc_args[@]}" 2>&1 | tee "$LOG_DIR/${BC_RUN_NAME}.log"
fi

echo "Done. Logs: $LOG_DIR"
