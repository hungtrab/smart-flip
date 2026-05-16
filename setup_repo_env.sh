#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:?VENV_DIR is required}"
REQ_FILE="${REQ_FILE:?REQ_FILE is required}"
TORCH_SPEC="${TORCH_SPEC:-torch}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
CLEAR_VENV="${CLEAR_VENV:-1}"
CLEAR_REPO_CACHE="${CLEAR_REPO_CACHE:-0}"
CLEAR_GLOBAL_HF_DATASETS_CACHE="${CLEAR_GLOBAL_HF_DATASETS_CACHE:-0}"

if [ ! -f "$REQ_FILE" ]; then
  echo "Missing requirements file: $REQ_FILE" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ "$CLEAR_VENV" = "1" ] && [ -d "$VENV_DIR" ]; then
  rm -rf "$VENV_DIR"
fi

if [ "$CLEAR_REPO_CACHE" = "1" ]; then
  rm -rf \
    "$REPO_DIR/data/cache" \
    "$REPO_DIR/results/eval" \
    "$REPO_DIR/results/eval/lm_eval"
fi

if [ "$CLEAR_GLOBAL_HF_DATASETS_CACHE" = "1" ]; then
  HF_CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}"
  rm -rf "$HF_CACHE_ROOT/datasets"
  if [ -d "$HF_CACHE_ROOT/hub" ]; then
    find "$HF_CACHE_ROOT/hub" -maxdepth 1 -mindepth 1 -type d -name 'datasets--*' -exec rm -rf {} +
  fi
fi

mkdir -p "$REPO_DIR/data/cache" "$REPO_DIR/results/eval" "$REPO_DIR/results/models"

"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [ -n "$TORCH_INDEX_URL" ]; then
  python -m pip install --upgrade --index-url "$TORCH_INDEX_URL" "$TORCH_SPEC"
else
  python -m pip install --upgrade "$TORCH_SPEC"
fi

python -m pip install --upgrade -r "$REQ_FILE"

python - <<'PY'
import importlib

for name in ["torch", "datasets", "transformers", "accelerate", "huggingface_hub", "peft", "lm_eval", "wandb"]:
    module = importlib.import_module(name)
    print(f"{name}=={getattr(module, '__version__', 'n/a')}")
PY

cat <<EOF

Setup complete.
Repo: $REPO_DIR
Venv: $VENV_DIR
Requirements: $REQ_FILE
Torch spec: $TORCH_SPEC
Torch index: ${TORCH_INDEX_URL:-<default index>}

Activate with:
  source "$VENV_DIR/bin/activate"

Run from:
  cd "$REPO_DIR"
EOF
