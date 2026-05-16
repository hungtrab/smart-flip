#!/usr/bin/env bash
set -u

# Preflight checks for the 3-bit EGBC C4/+BC runs.
# This script does not load or quantize models.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
GPU_LIST="${GPU_LIST:-$GPU,$GPU,$GPU,$GPU}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-$REPO_DIR/results/models}"
RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-$REPO_DIR/results/eval}"
CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-$REPO_DIR/data/cache/calibration}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-$REPO_DIR/data/cache/eval}"
WANDB_PROJECT="${WANDB_PROJECT:-egbc_c4_b3}"
RUN_EGBC_BEST="${RUN_EGBC_BEST:-1}"
RUN_BC_BASELINE="${RUN_BC_BASELINE:-1}"
RUN_FQ_RAW_ONCE="${RUN_FQ_RAW_ONCE:-1}"

STATUS=0

pass() {
  printf '[OK] %s\n' "$1"
}

fail() {
  printf '[FAIL] %s\n' "$1" >&2
  STATUS=1
}

warn() {
  printf '[WARN] %s\n' "$1" >&2
}

check_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    pass "file exists: $path"
  else
    fail "missing file: $path"
  fi
}

check_writable_dir() {
  local dir="$1"
  mkdir -p "$dir" 2>/dev/null
  if [[ -d "$dir" && -w "$dir" ]]; then
    pass "writable dir: $dir"
  else
    fail "not writable: $dir"
  fi
}

echo "================================================================"
echo "Preflight: egbc_c4_b3"
echo "Repo: $REPO_DIR"
echo "Python: $PYTHON_BIN"
echo "GPU_LIST: $GPU_LIST"
echo "W&B project: $WANDB_PROJECT"
echo "================================================================"

cd "$REPO_DIR" || exit 1

check_file "$REPO_DIR/main.py"
check_file "$SCRIPT_DIR/egbc_c4_b3.sh"
check_file "$SCRIPT_DIR/egbc_c4_b3_sequential.sh"

if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  pass "python command found: $PYTHON_BIN"
else
  fail "python command not found: $PYTHON_BIN"
fi

for script in \
  "$SCRIPT_DIR/egbc_c4_b3.sh" \
  "$SCRIPT_DIR/egbc_c4_b3_sequential.sh" \
  "$SCRIPT_DIR/egbc_c4_b3_validate.sh" \
  "$SCRIPT_DIR"/bias_correction/awq/*.sh \
  "$SCRIPT_DIR"/bias_correction/flatquant/*.sh \
  "$SCRIPT_DIR"/bias_correction/gptq/*.sh
do
  if bash -n "$script" >/dev/null 2>&1; then
    pass "bash syntax: ${script#$REPO_DIR/}"
  else
    fail "bash syntax error: ${script#$REPO_DIR/}"
  fi
done

check_writable_dir "$RESULTS_MODELS_DIR"
check_writable_dir "$RESULTS_EVAL_DIR"
check_writable_dir "$CALIBRATION_CACHE_DIR"
check_writable_dir "$EVAL_CACHE_DIR"
check_writable_dir "$REPO_DIR/logs/egbc_c4_b3"

if [[ -d "$MODELS_ROOT" ]]; then
  pass "MODELS_ROOT exists: $MODELS_ROOT"
else
  warn "MODELS_ROOT does not exist: $MODELS_ROOT; model ids will be downloaded from Hugging Face if tokens/access are valid"
fi

if [[ -f "$REPO_DIR/.env" ]]; then
  pass ".env exists"
else
  warn ".env not found; relying on exported HF_TOKEN/WANDB_API_KEY"
fi

"$PYTHON_BIN" - <<'PY'
import os
import sys
from pathlib import Path

status = 0

def ok(msg):
    print(f"[OK] {msg}")

def fail(msg):
    global status
    print(f"[FAIL] {msg}", file=sys.stderr)
    status = 1

def warn(msg):
    print(f"[WARN] {msg}", file=sys.stderr)

env_path = Path(".env")
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        ok("python-dotenv loaded .env")
    except Exception as exc:
        fail(f"could not load .env via python-dotenv: {exc}")

required_modules = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "huggingface_hub",
    "lm_eval",
    "wandb",
    "dotenv",
]

for name in required_modules:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "n/a")
        ok(f"import {name}=={version}")
    except Exception as exc:
        fail(f"cannot import {name}: {exc}")

hf_token = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACE_TOKEN")
)
if hf_token:
    ok("HF token is set")
else:
    fail("HF_TOKEN/HUGGINGFACE_HUB_TOKEN is not set")

if os.getenv("WANDB_API_KEY"):
    ok("WANDB_API_KEY is set")
else:
    fail("WANDB_API_KEY is not set")

try:
    import torch
    if torch.cuda.is_available():
        ok(f"CUDA available: {torch.cuda.device_count()} GPU(s), torch CUDA={torch.version.cuda}")
    else:
        fail("torch.cuda.is_available() is false")
except Exception as exc:
    fail(f"CUDA check failed: {exc}")

try:
    import contextlib
    import io
    import main
    parser = main.build_parser()
    with contextlib.redirect_stdout(io.StringIO()):
        parser.parse_args(["quantize", "--help"])
except SystemExit as exc:
    if exc.code == 0:
        ok("main.py parser builds")
    else:
        fail(f"main.py parser exited unexpectedly: {exc.code}")
except Exception as exc:
    fail(f"main.py parser check failed: {exc}")

if os.getenv("CHECK_HF_MODELS", "0") == "1" and hf_token:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        for model_id in [
            "Qwen/Qwen2.5-7B",
            "meta-llama/Meta-Llama-3.1-8B",
            "meta-llama/Meta-Llama-3-8B",
        ]:
            api.model_info(model_id)
            ok(f"Hugging Face model access: {model_id}")
    except Exception as exc:
        fail(f"Hugging Face model access check failed: {exc}")
else:
    warn("skipping Hugging Face API model access check; set CHECK_HF_MODELS=1 to enable")

sys.exit(status)
PY
PY_STATUS=$?
if [[ "$PY_STATUS" -ne 0 ]]; then
  STATUS=1
fi

FQ_RAW_DIR="$RESULTS_MODELS_DIR/flatquant_raw/flatquant_raw_Qwen2.5-7B"
if [[ "$RUN_BC_BASELINE" == "1" && "$RUN_EGBC_BEST" != "1" && "$RUN_FQ_RAW_ONCE" != "1" ]]; then
  if [[ -f "$FQ_RAW_DIR/flat_parameters.pth" ]]; then
    pass "FlatQuant raw artifact exists: $FQ_RAW_DIR/flat_parameters.pth"
  else
    fail "FlatQuant raw artifact missing: $FQ_RAW_DIR/flat_parameters.pth; use RUN_EGBC_BEST=1 RUN_FQ_RAW_ONCE=1 first"
  fi
else
  warn "FlatQuant raw artifact will be generated/reused during the sequential run if needed"
fi

echo "================================================================"
if [[ "$STATUS" -eq 0 ]]; then
  echo "Preflight passed."
else
  echo "Preflight failed. Fix [FAIL] items before running jobs." >&2
fi
echo "================================================================"

exit "$STATUS"
