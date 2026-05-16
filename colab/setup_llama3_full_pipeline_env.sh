#!/usr/bin/env bash
set -euo pipefail

# Colab environment setup for the isolated Llama3 full pipeline repo.
# Default target:
#   /content/smart-flip/for_llama3/smart-flip
#
# Usage on Colab:
#   cd /content/smart-flip
#   bash colab/setup_llama3_full_pipeline_env.sh
#
# Optional overrides:
#   LLAMA3_REPO_DIR=/content/smart-flip/for_llama3/smart-flip \
#   VENV_DIR=/content/smart-flip/for_llama3/smart-flip/.venv \
#   bash colab/setup_llama3_full_pipeline_env.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DEFAULT_LLAMA3_REPO_DIR="/content/smart-flip/for_llama3/smart-flip"
if [[ -d "$DEFAULT_LLAMA3_REPO_DIR" ]]; then
  LLAMA3_REPO_DIR="${LLAMA3_REPO_DIR:-$DEFAULT_LLAMA3_REPO_DIR}"
else
  LLAMA3_REPO_DIR="${LLAMA3_REPO_DIR:-$ROOT_DIR/for_llama3/smart-flip}"
fi

VENV_DIR="${VENV_DIR:-$LLAMA3_REPO_DIR/.venv}"
REQ_FILE="${REQ_FILE:-$LLAMA3_REPO_DIR/requirements_llama3.txt}"
PYTHON_FOR_UV="${PYTHON_FOR_UV:-python3}"
GET_PIP_URL="${GET_PIP_URL:-https://bootstrap.pypa.io/get-pip.py}"
GET_PIP_PATH="${GET_PIP_PATH:-/tmp/get-pip.py}"
RESET_VENV="${RESET_VENV:-0}"

if [[ ! -d "$LLAMA3_REPO_DIR" ]]; then
  echo "Missing Llama3 repo dir: $LLAMA3_REPO_DIR" >&2
  exit 1
fi

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Missing requirements file: $REQ_FILE" >&2
  exit 1
fi

cd "$LLAMA3_REPO_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing uv without pip..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv install failed or uv is not on PATH" >&2
  exit 1
fi

echo "Creating venv with uv: $VENV_DIR"
if [[ "$RESET_VENV" == "1" && -d "$VENV_DIR" ]]; then
  echo "RESET_VENV=1, removing existing venv: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi
uv venv --system-site-packages --python "$PYTHON_FOR_UV" "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Bootstrapping pip inside venv..."
curl -sS "$GET_PIP_URL" -o "$GET_PIP_PATH"
python "$GET_PIP_PATH"

echo "Installing dependencies into: $(python -c 'import sys; print(sys.executable)')"
python -m pip install --upgrade pip setuptools wheel

# Keep Colab's preinstalled torch/CUDA stack. Do not let pip pull a new torch,
# nvidia-* or cuda-* wheel set into this venv.
python -m pip install --no-cache-dir \
  numpy safetensors sentencepiece protobuf tiktoken pyarrow \
  accelerate peft tqdm psutil wandb python-dotenv evaluate scikit-learn \
  rouge-score sacrebleu sqlitedict pytablewriter word2number more-itertools

# Pin the old lm-eval stack used by this repo. Install these without deps so
# pip does not upgrade tokenizers, torch, CUDA, or Hugging Face packages behind
# our back. In particular, transformers 4.36.x needs tokenizers < 0.19.
python -m pip install --no-cache-dir --force-reinstall --no-deps \
  "transformers==4.36.0" \
  "tokenizers==0.15.2" \
  "datasets==2.17.1" \
  "lm-eval==0.4.4" \
  "huggingface-hub==0.24.6" \
  "fsspec==2023.10.0" \
  "dill==0.3.8" \
  "multiprocess==0.70.16" \
  "xxhash"

echo "Verifying environment..."
python - <<'PY'
import sys
import numpy
import torch
import datasets
import transformers
import lm_eval
import wandb

print("python", sys.executable)
print("numpy", numpy.__version__, numpy.__file__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("datasets", datasets.__version__)
print("transformers", transformers.__version__)
print("lm_eval", getattr(lm_eval, "__version__", "unknown"))
print("wandb", wandb.__version__)
PY

cat <<EOF

Environment ready.

Activate it later with:
  cd "$LLAMA3_REPO_DIR"
  source "$VENV_DIR/bin/activate"

Run Llama3 FlatQuant + EGBC tune:
  PYTHON_BIN="$VENV_DIR/bin/python" \\
  WANDB_PROJECT=egbc_tune_llama3_b3 GPU=0 \\
  bash scripts/bash/llama3_flatquant_b3_tune_egbc.sh

EOF
