#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

BITS="${1:-4}"

if [[ "$BITS" != "3" && "$BITS" != "4" ]]; then
  echo "Usage: bash scripts/bash/ablation/run_llama31_awq_ablation.sh [3|4]" >&2
  exit 2
fi

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B}"
ABLATION_VARIANT="${ABLATION_VARIANT:-no_k}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CALIB_DATASET="${CALIB_DATASET:-c4}"
N_CALIB="${N_CALIB:-128}"
CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
GROUP_SIZE="${GROUP_SIZE:-128}"
MAX_TOKENS_PER_SAMPLE="${MAX_TOKENS_PER_SAMPLE:-2048}"
LAYER_BATCH_SIZE="${LAYER_BATCH_SIZE:-16}"
LMHEAD_CHUNKS="${LMHEAD_CHUNKS:-4}"
SEED="${SEED:-42}"
STRIDE="${STRIDE:-512}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
C4_SAMPLES="${C4_SAMPLES:-500}"
RESULTS_MODELS_ROOT="${RESULTS_MODELS_ROOT:-${RESULTS_MODELS_DIR:-./results/models/ablation/llama31_awq_b${BITS}}}"
RESULTS_EVAL_ROOT="${RESULTS_EVAL_ROOT:-${RESULTS_EVAL_DIR:-./results/eval/ablation/llama31_awq_b${BITS}}}"
CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-./data/cache/calibration}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./data/cache/eval}"
WANDB_PROJECT="${WANDB_PROJECT:-smartflip_ablation}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
USE_WANDB="${USE_WANDB:-1}"

BEST_KNEE="${BEST_KNEE:-0.02}"
BEST_MAX_FLIP="${BEST_MAX_FLIP:-0.01}"
LM_EVAL_TASKS=(
  arc_challenge
  arc_easy
  boolq
  piqa
  rte
)

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --origin-method awq
  --post-correction smart_flip
  --bits "$BITS"
  --group-size "$GROUP_SIZE"
  --calib-dataset "$CALIB_DATASET"
  --n-calib "$N_CALIB"
  --calib-seqlen "$CALIB_SEQLEN"
  --max-tokens-per-sample "$MAX_TOKENS_PER_SAMPLE"
  --layer-batch-size "$LAYER_BATCH_SIZE"
  --lmhead-chunks "$LMHEAD_CHUNKS"
  --knee-tolerance "$BEST_KNEE"
  --max-flip-percent "$BEST_MAX_FLIP"
  --calibration-cache-dir "$CALIBRATION_CACHE_DIR"
  --eval-cache-dir "$EVAL_CACHE_DIR"
  --seed "$SEED"
  --stride "$STRIDE"
  --max-length "$MAX_LENGTH"
  --c4-samples "$C4_SAMPLES"
  --lm-eval-tasks "${LM_EVAL_TASKS[@]}"
)

if [[ "$USE_WANDB" == "1" ]]; then
  COMMON_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")
  if [[ -n "$WANDB_ENTITY" ]]; then
    COMMON_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

run_variant() {
  local requested_variant="$1"
  local variant
  local -a variant_args

  case "$requested_variant" in
    best)
      variant="best"
      variant_args=()
      ;;
    no_k|no_knee)
      variant="no_k"
      variant_args=(--disable-knee-mask)
      ;;
    no_f|no_max_flip)
      variant="no_f"
      variant_args=(--disable-max-flip-cap)
      ;;
    *)
      echo "Unknown ablation variant: $requested_variant. Use best, no_k, no_f, or all." >&2
      exit 2
      ;;
  esac

  local run_name="awq_ablation_llama31_b${BITS}_${variant}"
  local results_models_dir="${RESULTS_MODELS_ROOT}/${variant}"
  local results_eval_dir="${RESULTS_EVAL_ROOT}/${variant}"

  echo "==> ${run_name}"
  "$PYTHON_BIN" main.py quantize \
    "${COMMON_ARGS[@]}" \
    --results-models-dir "$results_models_dir" \
    --results-eval-dir "$results_eval_dir" \
    --run-name "$run_name" \
    --wandb-tags ablation "bits:${BITS}" "variant:${variant}" \
    "${variant_args[@]}"
}

if [[ "$ABLATION_VARIANT" == "all" ]]; then
  run_variant best
  run_variant no_k
  run_variant no_f
else
  run_variant "$ABLATION_VARIANT"
fi
