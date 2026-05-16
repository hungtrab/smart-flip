#!/usr/bin/env bash
set -euo pipefail

# Run 3-bit EGBC C4 checks and 3-bit +BC baselines.
# RUN_EGBC_BEST uses exact tuned configs from tune_best.
# RUN_EGBC_SELECTED uses untuned selected/default configs from the selected-run scripts.
# +BC intentionally uses method defaults; it is not tuned with Smart-Flip params.

GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -ra _GPUS <<< "$GPU_LIST"
GPU_QWEN_AWQ="${_GPUS[0]:-0}"
GPU_QWEN_FQ="${_GPUS[1]:-1}"
GPU_LLAMA31_AWQ="${_GPUS[2]:-2}"
GPU_LLAMA3_AWQ="${_GPUS[3]:-3}"
GPU_MISTRAL_AWQ="${_GPUS[4]:-${_GPUS[0]:-0}}"
GPU_MISTRAL_FQ="${_GPUS[5]:-${_GPUS[1]:-1}}"
GPU_LLAMA31_FQ="${_GPUS[6]:-${_GPUS[2]:-2}}"
GPU_LLAMA3_FQ="${_GPUS[7]:-${_GPUS[3]:-3}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-./results/models}"
RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-./results/eval}"
CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-./data/cache/calibration}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./data/cache/eval}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./data/cache/hf_datasets/datasets-egbc-c4-b3}"
LOG_DIR="${LOG_DIR:-./logs/egbc_c4_b3}"

WANDB_PROJECT="${WANDB_PROJECT:-egbc_c4_b3}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

BITS="3"
N_CALIB="${N_CALIB:-128}"
CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
CALIB_DATASET="${CALIB_DATASET:-c4}"
SEED="${SEED:-42}"
STRIDE="${STRIDE:-512}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
C4_SAMPLES="${C4_SAMPLES:-500}"
BIAS_CORRECTION_SAMPLES="${BIAS_CORRECTION_SAMPLES:-4096}"
LM_EVAL_TASKS=(arc_challenge arc_easy boolq piqa rte)

RUN_EGBC_BEST="${RUN_EGBC_BEST:-0}"
RUN_EGBC_SELECTED="${RUN_EGBC_SELECTED:-1}"
RUN_BC_BASELINE="${RUN_BC_BASELINE:-0}"
RUN_BC_EXTRA="${RUN_BC_EXTRA:-1}"
RUN_FLOAT_MODEL="${RUN_FLOAT_MODEL:-0}"
RUN_FQ_RAW="${RUN_FQ_RAW:-0}"

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

mkdir -p "$LOG_DIR" "$HF_DATASETS_CACHE"

wandb_args=()
if [[ -n "$WANDB_ENTITY" ]]; then
  wandb_args+=(--wandb-entity "$WANDB_ENTITY")
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

common_quant_args() {
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
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    "${wandb_args[@]}"
}

read_common_args() {
  local model_path="$1"
  local -n out_ref=$2
  mapfile -d '' -t out_ref < <(common_quant_args "$model_path")
}

flatquant_raw_dir() {
  local model_path="$1"
  local model_slug="${model_path##*/}"
  printf '%s/flatquant_raw/flatquant_raw_%s' "$RESULTS_MODELS_DIR" "$model_slug"
}

maybe_run_flatquant_raw() {
  local model_path="$1"
  local gpu="$2"
  local raw_dir
  raw_dir="$(flatquant_raw_dir "$model_path")"

  if [[ "$RUN_FQ_RAW" != "1" ]]; then
    if [[ ! -f "$raw_dir/flat_parameters.pth" ]]; then
      echo "Missing FlatQuant raw artifact: $raw_dir/flat_parameters.pth" >&2
      echo "Set RUN_FQ_RAW=1 to regenerate it, or set RESULTS_MODELS_DIR to the tune output root." >&2
      return 1
    fi
    return 0
  fi

  local args=()
  read_common_args "$model_path" args
  add_flatquant_args args

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" main.py quantize \
    "${args[@]}" \
    --origin-method flatquant \
    --post-correction none \
    --no-lm-eval \
    --run-name "flatquant_raw_${model_path##*/}"
}

run_egbc_awq_c4() {
  local model_path="$1"
  local gpu="$2"
  local knee="$3"
  local flip="$4"
  local model_slug="${model_path##*/}"
  local args=()
  read_common_args "$model_path" args

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" main.py quantize \
    "${args[@]}" \
    --origin-method awq \
    --post-correction smart_flip \
    --knee-tolerance "$knee" \
    --max-flip-percent "$flip" \
    --no-lm-eval \
    --run-name "c4_awq_sf_${model_slug}_b${BITS}_k${knee}_f${flip}"
}

run_egbc_flatquant_c4() {
  local model_path="$1"
  local gpu="$2"
  local knee="$3"
  local flip="$4"
  local model_slug="${model_path##*/}"
  local raw_dir
  raw_dir="$(flatquant_raw_dir "$model_path")"
  maybe_run_flatquant_raw "$model_path" "$gpu"

  local args=()
  read_common_args "$model_path" args
  add_flatquant_args args

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" main.py quantize \
    "${args[@]}" \
    --origin-method flatquant \
    --post-correction smart_flip \
    --knee-tolerance "$knee" \
    --max-flip-percent "$flip" \
    --flatquant-raw-path "$raw_dir" \
    --no-lm-eval \
    --run-name "c4_fq_sf_${model_slug}_b${BITS}_k${knee}_f${flip}"
}

run_bc_awq_full() {
  local model_path="$1"
  local gpu="$2"
  local model_slug="${model_path##*/}"
  local args=()
  read_common_args "$model_path" args

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" main.py quantize \
    "${args[@]}" \
    --origin-method awq \
    --post-correction bias_correction \
    --bias-correction-samples "$BIAS_CORRECTION_SAMPLES" \
    --lm-eval-tasks "${LM_EVAL_TASKS[@]}" \
    --run-name "awq_bc_${model_slug}_b${BITS}_full"
}

run_bc_flatquant_full() {
  local model_path="$1"
  local gpu="$2"
  local model_slug="${model_path##*/}"
  local raw_dir
  raw_dir="$(flatquant_raw_dir "$model_path")"
  maybe_run_flatquant_raw "$model_path" "$gpu"

  local args=()
  read_common_args "$model_path" args
  add_flatquant_args args

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" main.py quantize \
    "${args[@]}" \
    --origin-method flatquant \
    --post-correction bias_correction \
    --bias-correction-samples "$BIAS_CORRECTION_SAMPLES" \
    --flatquant-raw-path "$raw_dir" \
    --lm-eval-tasks "${LM_EVAL_TASKS[@]}" \
    --run-name "flatquant_bc_${model_slug}_b${BITS}_full"
}

run_float_full() {
  local model_path="$1"
  local gpu="$2"
  local model_slug="${model_path##*/}"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" main.py float_model \
    --model-path "$model_path" \
    --models-root "$MODELS_ROOT" \
    --results-eval-dir "$RESULTS_EVAL_DIR" \
    --eval-cache-dir "$EVAL_CACHE_DIR" \
    --seed "$SEED" \
    --stride "$STRIDE" \
    --max-length "$MAX_LENGTH" \
    --c4-samples "$C4_SAMPLES" \
    --lm-eval-tasks "${LM_EVAL_TASKS[@]}" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    "${wandb_args[@]}" \
    --run-name "float_${model_slug}_full"
}

run_worker() {
  case "$1" in
    # Exact tuned best configs extracted from tune_best / tuning CSV.
    qwen_awq_egbc_c4) run_egbc_awq_c4 "Qwen/Qwen2.5-7B" "$GPU_QWEN_AWQ" "0.02" "0.03" ;;
    qwen_fq_egbc_c4) run_egbc_flatquant_c4 "Qwen/Qwen2.5-7B" "$GPU_QWEN_FQ" "0.02" "0.04" ;;
    llama31_awq_egbc_c4) run_egbc_awq_c4 "meta-llama/Meta-Llama-3.1-8B" "$GPU_LLAMA31_AWQ" "0.05" "0.03" ;;
    llama3_awq_egbc_c4) run_egbc_awq_c4 "meta-llama/Meta-Llama-3-8B" "$GPU_LLAMA3_AWQ" "0" "0.01" ;;

    # Untuned selected/default configs from run_awq_selected_egbc_b3.sh.
    mistral_awq_selected_egbc_c4) run_egbc_awq_c4 "mistralai/Mistral-7B-v0.3" "$GPU_MISTRAL_AWQ" "0.02" "0.02" ;;
    llama31_awq_selected_egbc_c4) run_egbc_awq_c4 "meta-llama/Meta-Llama-3.1-8B" "$GPU_LLAMA31_AWQ" "0.02" "0.01" ;;
    qwen_awq_selected_egbc_c4) run_egbc_awq_c4 "Qwen/Qwen2.5-7B" "$GPU_QWEN_AWQ" "0.03" "0.02" ;;
    llama3_awq_selected_egbc_c4) run_egbc_awq_c4 "meta-llama/Meta-Llama-3-8B" "$GPU_LLAMA3_AWQ" "0.01" "0.05" ;;

    # Untuned selected/default configs from run_flatquant_selected_egbc_b3.sh
    # and run_flatquant_llama3_egbc_b3.sh.
    mistral_fq_selected_egbc_c4) run_egbc_flatquant_c4 "mistralai/Mistral-7B-v0.3" "$GPU_MISTRAL_FQ" "0.0" "0.05" ;;
    llama31_fq_selected_egbc_c4) run_egbc_flatquant_c4 "meta-llama/Meta-Llama-3.1-8B" "$GPU_LLAMA31_FQ" "0.0" "0.05" ;;
    qwen_fq_selected_egbc_c4) run_egbc_flatquant_c4 "Qwen/Qwen2.5-7B" "$GPU_QWEN_FQ" "0.01" "0.05" ;;
    llama3_fq_selected_egbc_c4) run_egbc_flatquant_c4 "meta-llama/Meta-Llama-3-8B" "$GPU_LLAMA3_FQ" "0.02" "0.05" ;;

    # +BC baselines: default method parameters, full eval = 5 challenges + WikiText-2 + C4.
    qwen_awq_bc) run_bc_awq_full "Qwen/Qwen2.5-7B" "$GPU_QWEN_AWQ" ;;
    qwen_fq_bc) run_bc_flatquant_full "Qwen/Qwen2.5-7B" "$GPU_QWEN_FQ" ;;
    llama31_awq_bc) run_bc_awq_full "meta-llama/Meta-Llama-3.1-8B" "$GPU_LLAMA31_AWQ" ;;
    llama3_awq_bc) run_bc_awq_full "meta-llama/Meta-Llama-3-8B" "$GPU_LLAMA3_AWQ" ;;
    mistral_awq_bc) run_bc_awq_full "mistralai/Mistral-7B-v0.3" "$GPU_MISTRAL_AWQ" ;;
    mistral_fq_bc) run_bc_flatquant_full "mistralai/Mistral-7B-v0.3" "$GPU_MISTRAL_FQ" ;;
    llama31_fq_bc) run_bc_flatquant_full "meta-llama/Meta-Llama-3.1-8B" "$GPU_LLAMA31_FQ" ;;
    llama3_fq_bc) run_bc_flatquant_full "meta-llama/Meta-Llama-3-8B" "$GPU_LLAMA3_FQ" ;;
    qwen_float) run_float_full "Qwen/Qwen2.5-7B" "$GPU_QWEN_AWQ" ;;
    llama31_float) run_float_full "meta-llama/Meta-Llama-3.1-8B" "$GPU_LLAMA31_AWQ" ;;
    llama3_float) run_float_full "meta-llama/Meta-Llama-3-8B" "$GPU_LLAMA3_AWQ" ;;
    *) echo "Unknown WORKER: $1" >&2; return 2 ;;
  esac
}

run_phase() {
  local phase_name="$1"
  shift
  local status=0
  local pids=()
  local names=()

  echo "==> phase: $phase_name"
  for worker in "$@"; do
    echo "  starting $worker"
    run_worker "$worker" > "$LOG_DIR/${worker}.log" 2>&1 &
    pids+=("$!")
    names+=("$worker")
  done

  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "FAILED: ${names[$i]} (log: $LOG_DIR/${names[$i]}.log)" >&2
      status=1
    fi
  done
  return "$status"
}

if [[ -n "${WORKER:-}" ]]; then
  run_worker "$WORKER" 2>&1 | tee "$LOG_DIR/${WORKER}.log"
  exit "${PIPESTATUS[0]}"
fi

echo "================================================================"
echo "egbc_c4_b3: project=$WANDB_PROJECT bits=$BITS c4_samples=$C4_SAMPLES"
echo "phases: best=$RUN_EGBC_BEST selected=$RUN_EGBC_SELECTED bc_baseline=$RUN_BC_BASELINE bc_extra=$RUN_BC_EXTRA float=$RUN_FLOAT_MODEL"
echo "logs: $LOG_DIR"
echo "================================================================"

if [[ "$RUN_FLOAT_MODEL" == "1" ]]; then
  run_phase float qwen_float llama31_float llama3_float
fi

if [[ "$RUN_EGBC_BEST" == "1" ]]; then
  run_phase egbc_best_c4 \
    qwen_awq_egbc_c4 \
    qwen_fq_egbc_c4 \
    llama31_awq_egbc_c4 \
    llama3_awq_egbc_c4
fi

if [[ "$RUN_EGBC_SELECTED" == "1" ]]; then
  run_phase egbc_selected_c4 \
    mistral_awq_selected_egbc_c4 \
    llama31_awq_selected_egbc_c4 \
    qwen_awq_selected_egbc_c4 \
    llama3_awq_selected_egbc_c4 \
    mistral_fq_selected_egbc_c4 \
    llama31_fq_selected_egbc_c4 \
    qwen_fq_selected_egbc_c4 \
    llama3_fq_selected_egbc_c4
fi

if [[ "$RUN_BC_BASELINE" == "1" ]]; then
  run_phase bc_baseline_full \
    qwen_awq_bc \
    qwen_fq_bc \
    llama31_awq_bc \
    llama3_awq_bc
fi

if [[ "$RUN_BC_EXTRA" == "1" ]]; then
  run_phase bc_extra_full \
    mistral_awq_bc \
    mistral_fq_bc \
    llama31_fq_bc \
    llama3_fq_bc
fi

echo "Done. Monitor or inspect logs in $LOG_DIR"
