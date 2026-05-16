#!/usr/bin/env bash
set -euo pipefail

# Repair the currently active Colab/Python environment without deleting caches
# or recreating the venv. Run this from the repo root after activating the env.

PYTHON_BIN="${PYTHON_BIN:-python}"
REQ_FILE="${REQ_FILE:-requirements.txt}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -f "$REQ_FILE" ]; then
  echo "Missing requirements file: $REQ_FILE" >&2
  exit 1
fi

echo "Using Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "Installing/repairing Python dependencies from $REQ_FILE"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install --upgrade -r "$REQ_FILE"

# Explicitly install packages that are easy to miss in partial Colab setups.
"$PYTHON_BIN" -m pip install --upgrade \
  numpy \
  safetensors \
  sentencepiece \
  protobuf \
  tiktoken \
  pyarrow

echo
echo "Verifying imports and CUDA..."
"$PYTHON_BIN" - <<'PY'
import importlib

modules = [
    "numpy",
    "torch",
    "datasets",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "peft",
    "lm_eval",
    "wandb",
    "dotenv",
    "safetensors",
]

for name in modules:
    module = importlib.import_module(name)
    print(f"{name}=={getattr(module, '__version__', 'n/a')}")

import torch
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device=", torch.cuda.get_device_name(0))
PY

cat <<'EOF'

Repair complete.

Rerun example:
  RUN_EGBC_SELECTED=0 RUN_BC_EXTRA=1 WANDB_PROJECT=egbc_c4_b3 GPU=0 bash scripts/bash/egbc_c4_b3_sequential.sh
EOF
