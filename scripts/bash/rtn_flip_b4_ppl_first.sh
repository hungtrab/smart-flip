#!/usr/bin/env bash
set -euo pipefail

# RTN vs RTN+Flip debug runner.
#
# Strategy:
#   1. Run cheap perplexity first for naive RTN and a Flip k/f grid.
#   2. Only run lm-eval for the best TOP_N Flip configs by PPL.
#
# This intentionally excludes Llama3 by default. Keep Llama3 in the isolated
# for_llama3 repo/session.

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-./results/models}"
RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-./results/eval}"
CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-./data/cache/calibration}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./data/cache/eval}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./data/cache/hf_datasets/datasets-rtn-flip-b4-ppl-first}"
LOG_DIR="${LOG_DIR:-./logs/rtn_flip_b4_ppl_first}"

WANDB_PROJECT="${WANDB_PROJECT:-rtn_flip_b4_ppl_first}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
USE_WANDB="${USE_WANDB:-1}"

BITS="${BITS:-4}"
FLATQUANT_A_BITS="${FLATQUANT_A_BITS:-16}"
ACT_SUFFIX="a${FLATQUANT_A_BITS}"
N_CALIB="${N_CALIB:-128}"
CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
CALIB_DATASET="${CALIB_DATASET:-c4}"
SEED="${SEED:-42}"
STRIDE="${STRIDE:-512}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
C4_SAMPLES="${C4_SAMPLES:-128}"
LM_EVAL_TASKS=(arc_challenge arc_easy boolq piqa rte)

KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)
MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)

# STAGE=ppl: run PPL sweep only.
# STAGE=lm_eval_top: read existing PPL JSONs, run lm-eval for top configs.
# STAGE=all: run PPL sweep, then lm-eval top configs.
STAGE="${STAGE:-ppl}"
TOP_N="${TOP_N:-3}"
SELECT_BY="${SELECT_BY:-C4}"
RUN_RAW="${RUN_RAW:-auto}"
RUN_RAW_LM_EVAL="${RUN_RAW_LM_EVAL:-1}"
SKIP_EXISTING_JSON="${SKIP_EXISTING_JSON:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-120}"
RUN_LLAMA3="${RUN_LLAMA3:-0}"

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
  local include_lm_eval="$2"
  local -n out_ref=$3

  out_ref=(
    --model-path "$model_path"
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
    --flatquant-a-bits "$FLATQUANT_A_BITS"
    --include-c4
    --lm-eval-tasks "${LM_EVAL_TASKS[@]}"
    "${wandb_args[@]}"
  )

  if [[ "$include_lm_eval" == "1" ]]; then
    out_ref+=(--include-lm-eval)
  else
    out_ref+=(--no-lm-eval)
  fi
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

run_raw_ppl() {
  local model_path="$1"
  local slug="${model_path##*/}"
  local run_name="rtn_raw_${slug}_b${BITS}_${ACT_SUFFIX}_ppl"
  local args=()

  local run_raw=false
  if [[ "$RUN_RAW" == "1" ]]; then
    run_raw=true
  elif [[ "$RUN_RAW" == "auto" && ! -f "$RESULTS_EVAL_DIR/${run_name}.json" ]]; then
    run_raw=true
  fi

  if [[ "$run_raw" != "true" ]]; then
    echo "Skipping raw PPL; found or disabled: $run_name"
    return 0
  fi

  common_args "$model_path" 0 args
  add_naive_rtn_args args
  run_with_retry "$run_name" \
    "${args[@]}" \
    --origin-method flatquant \
    --post-correction none \
    --run-name "$run_name"
}

run_flip_ppl_grid() {
  local model_path="$1"
  local slug="${model_path##*/}"
  local args=()

  for knee in "${KNEE_VALUES[@]}"; do
    for flip in "${MAX_FLIP_VALUES[@]}"; do
      local run_name="rtn_flip_${slug}_b${BITS}_${ACT_SUFFIX}_k${knee}_f${flip}_ppl"
      common_args "$model_path" 0 args
      add_naive_rtn_args args
      run_with_retry "$run_name" \
        "${args[@]}" \
        --origin-method flatquant \
        --post-correction smart_flip \
        --knee-tolerance "$knee" \
        --max-flip-percent "$flip" \
        --run-name "$run_name"
    done
  done
}

select_top_configs() {
  local slug="$1"
  python3 - "$RESULTS_EVAL_DIR" "$slug" "$BITS" "$ACT_SUFFIX" "$TOP_N" "$SELECT_BY" <<'PY'
import glob
import json
import math
import os
import re
import sys

results_dir, slug, bits, act_suffix, top_n, select_by = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
pattern = os.path.join(results_dir, f"rtn_flip_{slug}_b{bits}_{act_suffix}_k*_f*_ppl.json")
items = []

def first_ppl(payload, dataset):
    variants = payload.get("perplexity", {}).get(dataset, {})
    if not isinstance(variants, dict):
        return None
    for metrics in variants.values():
        if isinstance(metrics, dict) and isinstance(metrics.get("perplexity"), (int, float)):
            value = float(metrics["perplexity"])
            if math.isfinite(value):
                return value
    return None

for path in glob.glob(pattern):
    name = os.path.basename(path)
    match = re.search(r"_k([0-9.]+)_f([0-9.]+)_ppl\\.json$", name)
    if not match:
        continue
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    wiki = first_ppl(payload, "WikiText-2")
    c4 = first_ppl(payload, "C4")
    if select_by == "WikiText-2":
        score = wiki
    elif select_by == "sum":
        score = None if wiki is None or c4 is None else wiki + c4
    else:
        score = c4
    if score is None:
        continue
    items.append((score, match.group(1), match.group(2), wiki, c4, name))

items.sort(key=lambda x: x[0])
for score, knee, flip, wiki, c4, name in items[:top_n]:
    print(knee, flip, score, wiki, c4, name)
PY
}

run_raw_lm_eval() {
  local model_path="$1"
  local slug="${model_path##*/}"
  local run_name="rtn_raw_${slug}_b${BITS}_${ACT_SUFFIX}_lmeval"
  local args=()

  if [[ "$RUN_RAW_LM_EVAL" != "1" ]]; then
    return 0
  fi

  common_args "$model_path" 1 args
  add_naive_rtn_args args
  run_with_retry "$run_name" \
    "${args[@]}" \
    --origin-method flatquant \
    --post-correction none \
    --run-name "$run_name"
}

run_lm_eval_top() {
  local model_path="$1"
  local slug="${model_path##*/}"
  local args=()
  local selection

  run_raw_lm_eval "$model_path"

  selection="$(select_top_configs "$slug")"
  if [[ -z "$selection" ]]; then
    echo "No PPL sweep JSON found for $slug. Run STAGE=ppl first." >&2
    return 1
  fi

  echo "Top ${TOP_N} configs for $slug by ${SELECT_BY}:"
  echo "$selection"

  while read -r knee flip _score _wiki _c4 _name; do
    [[ -z "$knee" ]] && continue
    local run_name="rtn_flip_${slug}_b${BITS}_${ACT_SUFFIX}_k${knee}_f${flip}_lmeval"
    common_args "$model_path" 1 args
    add_naive_rtn_args args
    run_with_retry "$run_name" \
      "${args[@]}" \
      --origin-method flatquant \
      --post-correction smart_flip \
      --knee-tolerance "$knee" \
      --max-flip-percent "$flip" \
      --run-name "$run_name"
  done <<< "$selection"
}

for model in "${MODELS[@]}"; do
  echo "================================================================"
  echo "RTN PPL-first b${BITS}: $model"
  echo "stage=$STAGE select_by=$SELECT_BY top_n=$TOP_N a_bits=$FLATQUANT_A_BITS wandb_project=$WANDB_PROJECT"
  echo "================================================================"

  if [[ "$STAGE" == "ppl" || "$STAGE" == "all" ]]; then
    run_raw_ppl "$model"
    run_flip_ppl_grid "$model"
  fi

  if [[ "$STAGE" == "lm_eval_top" || "$STAGE" == "all" ]]; then
    run_lm_eval_top "$model"
  fi
done

echo "Done. Logs: $LOG_DIR"
