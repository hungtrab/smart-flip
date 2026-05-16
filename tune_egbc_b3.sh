#!/usr/bin/env bash
# tune_egbc_b3.sh
#
# Re-tune AWQ & FlatQuant (3-bit) với mục tiêu 5 lm_eval tasks:
#   arc_challenge  arc_easy  boolq  piqa  rte
#
# 4 workers chạy song song, mỗi worker trên 1 GPU:
#   Worker 0 → Qwen2.5-7B   AWQ
#   Worker 1 → Qwen2.5-7B   FlatQuant
#   Worker 2 → Llama3.1-8B  AWQ
#   Worker 3 → Llama3-8B    AWQ
#
# Cách dùng:
#   bash tune_egbc_b3.sh
#
# Override:
#   GPU_LIST=0,1,2,3          GPU_LIST=4,5,6,7 bash tune_egbc_b3.sh
#   MODELS_ROOT=/data/models
#   RUN_FLOAT_MODEL=0         # bỏ qua float eval
#   RUN_RAW_QUANTIZE=0        # bỏ qua raw, cần FQ_RAW_DIR
#   FQ_RAW_DIR=./results/models/flatquant_raw/flatquant_raw_Qwen2.5-7B
#
# Kết quả log: ./logs/tune_egbc_b3/<worker>.log
# wandb project: egbc_tune

set -uo pipefail

# ── GPU assignment ────────────────────────────────────────────────────────────
GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -ra _GPUS <<< "$GPU_LIST"
GPU_QWEN_AWQ="${_GPUS[0]:-0}"
GPU_QWEN_FQ="${_GPUS[1]:-1}"
GPU_LLAMA31_AWQ="${_GPUS[2]:-2}"
GPU_LLAMA3_AWQ="${_GPUS[3]:-3}"

# ── Paths ─────────────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-./results/models}"
RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-./results/eval}"
CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-./data/cache/calibration}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./data/cache/eval}"
LOG_DIR="${LOG_DIR:-./logs/tune_egbc_b3}"

# ── W&B ──────────────────────────────────────────────────────────────────────
WANDB_PROJECT="egbc_tune"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# ── Calibration & eval ────────────────────────────────────────────────────────
BITS="3"
N_CALIB=128
CALIB_SEQLEN=2048
SEED=42
STRIDE=512
MAX_LENGTH=2048
LM_EVAL_TASKS=(arc_challenge arc_easy boolq piqa rte)   # 5 target tasks

# ── Pipeline flags ────────────────────────────────────────────────────────────
RUN_FLOAT_MODEL="${RUN_FLOAT_MODEL:-1}"
RUN_RAW_QUANTIZE="${RUN_RAW_QUANTIZE:-1}"

# FlatQuant: path đến raw artifact đã có (chỉ cần nếu RUN_RAW_QUANTIZE=0)
FQ_RAW_DIR="${FQ_RAW_DIR:-}"

# ── Grid (6 × 5 = 30 combinations, dùng chung cho AWQ và FlatQuant) ───────────
KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)
MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)

# ── FlatQuant hyperparams ─────────────────────────────────────────────────────
FLATQUANT_EPOCHS="${FLATQUANT_EPOCHS:-15}"
FLATQUANT_CALI_BSZ="${FLATQUANT_CALI_BSZ:-4}"
FLATQUANT_LR="${FLATQUANT_LR:-5e-3}"
FLATQUANT_DIAG_INIT="${FLATQUANT_DIAG_INIT:-sq_style}"
FLATQUANT_DIAG_ALPHA="${FLATQUANT_DIAG_ALPHA:-0.3}"
FLATQUANT_CALI_TRANS="${FLATQUANT_CALI_TRANS:-1}"
FLATQUANT_ADD_DIAG="${FLATQUANT_ADD_DIAG:-1}"
FLATQUANT_LWC="${FLATQUANT_LWC:-1}"
FLATQUANT_LAC="${FLATQUANT_LAC:-1}"

mkdir -p "$LOG_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
# Worker: AWQ (float + raw + smart_flip grid)
# ═══════════════════════════════════════════════════════════════════════════════
run_awq() {
    local model_path="$1"
    local gpu="$2"
    local model_slug="${model_path##*/}"

    export CUDA_VISIBLE_DEVICES="$gpu"

    local EVAL_ARGS=(
        --models-root "$MODELS_ROOT"
        --results-eval-dir "$RESULTS_EVAL_DIR"
        --eval-cache-dir "$EVAL_CACHE_DIR"
        --seed "$SEED" --stride "$STRIDE" --max-length "$MAX_LENGTH"
        --lm-eval-tasks "${LM_EVAL_TASKS[@]}"
        --no-c4
        --use-wandb --wandb-project "$WANDB_PROJECT"
    )
    [[ -n "$WANDB_ENTITY" ]] && EVAL_ARGS+=(--wandb-entity "$WANDB_ENTITY")

    local QUANT_ARGS=(
        --model-path "$model_path"
        --models-root "$MODELS_ROOT"
        --results-models-dir "$RESULTS_MODELS_DIR"
        --results-eval-dir "$RESULTS_EVAL_DIR"
        --calibration-cache-dir "$CALIBRATION_CACHE_DIR"
        --eval-cache-dir "$EVAL_CACHE_DIR"
        --n-calib "$N_CALIB" --calib-seqlen "$CALIB_SEQLEN"
        --seed "$SEED" --stride "$STRIDE" --max-length "$MAX_LENGTH"
        --bits "$BITS"
        --lm-eval-tasks "${LM_EVAL_TASKS[@]}"
        --no-c4
        --use-wandb --wandb-project "$WANDB_PROJECT"
    )
    [[ -n "$WANDB_ENTITY" ]] && QUANT_ARGS+=(--wandb-entity "$WANDB_ENTITY")

    # float_model
    if [[ "$RUN_FLOAT_MODEL" == "1" ]]; then
        echo "[AWQ ${model_slug}] float_model"
        "$PYTHON_BIN" main.py float_model \
            --model-path "$model_path" \
            "${EVAL_ARGS[@]}" \
            --run-name "awq_float_${model_slug}"
    fi

    # raw quantize
    if [[ "$RUN_RAW_QUANTIZE" == "1" ]]; then
        echo "[AWQ ${model_slug}] raw quantize"
        "$PYTHON_BIN" main.py quantize \
            "${QUANT_ARGS[@]}" \
            --origin-method awq --post-correction none \
            --run-name "awq_raw_${model_slug}"
    fi

    # smart_flip grid
    for knee in "${KNEE_VALUES[@]}"; do
        for flip in "${MAX_FLIP_VALUES[@]}"; do
            echo "[AWQ ${model_slug}] smart_flip k=${knee} f=${flip}"
            "$PYTHON_BIN" main.py quantize \
                "${QUANT_ARGS[@]}" \
                --origin-method awq --post-correction smart_flip \
                --knee-tolerance "$knee" --max-flip-percent "$flip" \
                --run-name "awq_sf_${model_slug}_b${BITS}_k${knee}_f${flip}"
        done
    done

    echo "[AWQ ${model_slug}] DONE"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Worker: FlatQuant (float + raw + smart_flip grid)
# ═══════════════════════════════════════════════════════════════════════════════
run_flatquant() {
    local model_path="$1"
    local gpu="$2"
    local model_slug="${model_path##*/}"

    export CUDA_VISIBLE_DEVICES="$gpu"

    local EVAL_ARGS=(
        --models-root "$MODELS_ROOT"
        --results-eval-dir "$RESULTS_EVAL_DIR"
        --eval-cache-dir "$EVAL_CACHE_DIR"
        --seed "$SEED" --stride "$STRIDE" --max-length "$MAX_LENGTH"
        --lm-eval-tasks "${LM_EVAL_TASKS[@]}"
        --no-c4
        --use-wandb --wandb-project "$WANDB_PROJECT"
    )
    [[ -n "$WANDB_ENTITY" ]] && EVAL_ARGS+=(--wandb-entity "$WANDB_ENTITY")

    local FQ_ARGS=(
        --flatquant-epochs "$FLATQUANT_EPOCHS"
        --flatquant-cali-bsz "$FLATQUANT_CALI_BSZ"
        --flatquant-lr "$FLATQUANT_LR"
        --flatquant-diag-init "$FLATQUANT_DIAG_INIT"
        --flatquant-diag-alpha "$FLATQUANT_DIAG_ALPHA"
    )
    [[ "$FLATQUANT_CALI_TRANS" == "1" ]] && FQ_ARGS+=(--flatquant-cali-trans)  || FQ_ARGS+=(--no-flatquant-cali-trans)
    [[ "$FLATQUANT_ADD_DIAG"   == "1" ]] && FQ_ARGS+=(--flatquant-add-diag)    || FQ_ARGS+=(--no-flatquant-add-diag)
    [[ "$FLATQUANT_LWC"        == "1" ]] && FQ_ARGS+=(--flatquant-lwc)          || FQ_ARGS+=(--no-flatquant-lwc)
    [[ "$FLATQUANT_LAC"        == "1" ]] && FQ_ARGS+=(--flatquant-lac)          || FQ_ARGS+=(--no-flatquant-lac)

    local QUANT_BASE=(
        --model-path "$model_path"
        --models-root "$MODELS_ROOT"
        --results-models-dir "$RESULTS_MODELS_DIR"
        --results-eval-dir "$RESULTS_EVAL_DIR"
        --calibration-cache-dir "$CALIBRATION_CACHE_DIR"
        --eval-cache-dir "$EVAL_CACHE_DIR"
        --n-calib "$N_CALIB" --calib-seqlen "$CALIB_SEQLEN"
        --seed "$SEED" --stride "$STRIDE" --max-length "$MAX_LENGTH"
        --bits "$BITS"
        --lm-eval-tasks "${LM_EVAL_TASKS[@]}"
        --no-c4
        --use-wandb --wandb-project "$WANDB_PROJECT"
        "${FQ_ARGS[@]}"
    )
    [[ -n "$WANDB_ENTITY" ]] && QUANT_BASE+=(--wandb-entity "$WANDB_ENTITY")

    local raw_run_name="flatquant_raw_${model_slug}"
    local raw_dir="${FQ_RAW_DIR:-${RESULTS_MODELS_DIR}/flatquant_raw/${raw_run_name}}"

    # float_model
    if [[ "$RUN_FLOAT_MODEL" == "1" ]]; then
        echo "[FQ ${model_slug}] float_model"
        "$PYTHON_BIN" main.py float_model \
            --model-path "$model_path" \
            "${EVAL_ARGS[@]}" \
            --run-name "flatquant_float_${model_slug}"
    fi

    # raw quantize (thời gian dài ~2-3 giờ)
    if [[ "$RUN_RAW_QUANTIZE" == "1" ]]; then
        echo "[FQ ${model_slug}] raw quantize (epochs=${FLATQUANT_EPOCHS})"
        "$PYTHON_BIN" main.py quantize \
            "${QUANT_BASE[@]}" \
            --origin-method flatquant --post-correction none \
            --run-name "$raw_run_name"
    elif [[ ! -f "${raw_dir}/flat_parameters.pth" ]]; then
        echo "[FQ ${model_slug}] ERROR: flat_parameters.pth not found at ${raw_dir}" >&2
        echo "  Set RUN_RAW_QUANTIZE=1 or FQ_RAW_DIR=<path>" >&2
        return 1
    else
        echo "[FQ ${model_slug}] reusing raw artifact at ${raw_dir}"
    fi

    # smart_flip grid
    for knee in "${KNEE_VALUES[@]}"; do
        for flip in "${MAX_FLIP_VALUES[@]}"; do
            echo "[FQ ${model_slug}] smart_flip k=${knee} f=${flip}"
            "$PYTHON_BIN" main.py quantize \
                "${QUANT_BASE[@]}" \
                --origin-method flatquant --post-correction smart_flip \
                --knee-tolerance "$knee" --max-flip-percent "$flip" \
                --flatquant-raw-path "$raw_dir" \
                --run-name "fq_sf_${model_slug}_b${BITS}_k${knee}_f${flip}"
        done
    done

    echo "[FQ ${model_slug}] DONE"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Launch 4 workers
# ═══════════════════════════════════════════════════════════════════════════════
# ── Worker map (tên → function + model + gpu) ─────────────────────────────────
declare -A WORKER_GPU=(
    [qwen_awq]="$GPU_QWEN_AWQ"
    [qwen_fq]="$GPU_QWEN_FQ"
    [llama31_awq]="$GPU_LLAMA31_AWQ"
    [llama3_awq]="$GPU_LLAMA3_AWQ"
)

run_worker() {
    local name="$1"
    case "$name" in
        qwen_awq)    run_awq       "Qwen/Qwen2.5-7B"              "${WORKER_GPU[$name]}" ;;
        qwen_fq)     run_flatquant "Qwen/Qwen2.5-7B"              "${WORKER_GPU[$name]}" ;;
        llama31_awq) run_awq       "meta-llama/Meta-Llama-3.1-8B" "${WORKER_GPU[$name]}" ;;
        llama3_awq)  run_awq       "meta-llama/Meta-Llama-3-8B"   "${WORKER_GPU[$name]}" ;;
        *) echo "Unknown worker: $name. Choices: qwen_awq qwen_fq llama31_awq llama3_awq" >&2; exit 1 ;;
    esac
}

# ── Single-worker mode: WORKER=qwen_fq bash tune_egbc_b3.sh ──────────────────
if [[ -n "${WORKER:-}" ]]; then
    echo "Single-worker mode: ${WORKER} on GPU ${WORKER_GPU[$WORKER]}"
    run_worker "$WORKER" 2>&1 | tee "$LOG_DIR/${WORKER}.log"
    exit $?
fi

# ── All-workers mode ──────────────────────────────────────────────────────────
echo "================================================================"
echo "tune_egbc_b3.sh — target tasks: ${LM_EVAL_TASKS[*]}"
echo "wandb project: ${WANDB_PROJECT}"
echo "================================================================"
printf "  GPU %s → Qwen2.5-7B   AWQ        (log: %s/qwen_awq.log)\n"    "$GPU_QWEN_AWQ"    "$LOG_DIR"
printf "  GPU %s → Qwen2.5-7B   FlatQuant  (log: %s/qwen_fq.log)\n"     "$GPU_QWEN_FQ"     "$LOG_DIR"
printf "  GPU %s → Llama3.1-8B  AWQ        (log: %s/llama31_awq.log)\n" "$GPU_LLAMA31_AWQ" "$LOG_DIR"
printf "  GPU %s → Llama3-8B    AWQ        (log: %s/llama3_awq.log)\n"  "$GPU_LLAMA3_AWQ"  "$LOG_DIR"
echo

run_worker qwen_awq    >> "$LOG_DIR/qwen_awq.log"    2>&1 & PID_QWEN_AWQ=$!
run_worker qwen_fq     >> "$LOG_DIR/qwen_fq.log"     2>&1 & PID_QWEN_FQ=$!
run_worker llama31_awq >> "$LOG_DIR/llama31_awq.log" 2>&1 & PID_LLAMA31=$!
run_worker llama3_awq  >> "$LOG_DIR/llama3_awq.log"  2>&1 & PID_LLAMA3=$!

echo "Workers started — PIDs: qwen_awq=${PID_QWEN_AWQ} qwen_fq=${PID_QWEN_FQ} llama31=${PID_LLAMA31} llama3=${PID_LLAMA3}"
echo "Monitor: tail -f ${LOG_DIR}/*.log"
echo

STATUS=0
wait "$PID_QWEN_AWQ"   || { echo "FAILED: Qwen AWQ";        STATUS=1; }
wait "$PID_QWEN_FQ"    || { echo "FAILED: Qwen FlatQuant";  STATUS=1; }
wait "$PID_LLAMA31"    || { echo "FAILED: Llama3.1 AWQ";    STATUS=1; }
wait "$PID_LLAMA3"     || { echo "FAILED: Llama3-8B AWQ";   STATUS=1; }

echo
if [[ "$STATUS" -eq 0 ]]; then
    echo "================================================================"
    echo "All workers completed successfully."
    echo "================================================================"
else
    echo "================================================================"
    echo "Some workers FAILED. Check logs in ${LOG_DIR}/"
    echo "================================================================"
    exit 1
fi
