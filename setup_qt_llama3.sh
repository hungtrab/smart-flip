#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
QT_YML_PATH="${QT_YML_PATH:-}"
SANITIZED_QT_YML="${SANITIZED_QT_YML:-$REPO_DIR/qt.server.yml}"

ENV_NAME="${ENV_NAME:-qt-llama3}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$HOME/bin/micromamba}"
MICROMAMBA_PLATFORM="${MICROMAMBA_PLATFORM:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

CLEAR_ENV="${CLEAR_ENV:-1}"
CLEAR_REPO_CACHE="${CLEAR_REPO_CACHE:-1}"
CLEAR_GLOBAL_HF_DATASETS_CACHE="${CLEAR_GLOBAL_HF_DATASETS_CACHE:-1}"

if [ -z "$QT_YML_PATH" ]; then
  if [ -f "$REPO_DIR/qt.yml" ]; then
    QT_YML_PATH="$REPO_DIR/qt.yml"
  elif [ -f "$REPO_DIR/../qt.yml" ]; then
    QT_YML_PATH="$REPO_DIR/../qt.yml"
  fi
fi

if [ ! -f "$QT_YML_PATH" ]; then
  echo "Missing qt.yml: $QT_YML_PATH" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ -z "$MICROMAMBA_PLATFORM" ]; then
  case "$(uname -m)" in
    x86_64|amd64)
      MICROMAMBA_PLATFORM="linux-64"
      ;;
    aarch64|arm64)
      MICROMAMBA_PLATFORM="linux-aarch64"
      ;;
    *)
      echo "Unsupported architecture for micromamba bootstrap: $(uname -m)" >&2
      exit 1
      ;;
  esac
fi

mkdir -p "$(dirname "$MICROMAMBA_BIN")"
mkdir -p "$MAMBA_ROOT_PREFIX"

if [ ! -x "$MICROMAMBA_BIN" ]; then
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT
  archive_url="https://micro.mamba.pm/api/micromamba/${MICROMAMBA_PLATFORM}/latest"
  if command -v curl >/dev/null 2>&1; then
    curl -Ls "$archive_url" | tar -xj -C "$tmp_dir" bin/micromamba
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "$archive_url" | tar -xj -C "$tmp_dir" bin/micromamba
  else
    echo "Need curl or wget to bootstrap micromamba." >&2
    exit 1
  fi
  install -m 755 "$tmp_dir/bin/micromamba" "$MICROMAMBA_BIN"
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

"$PYTHON_BIN" - "$QT_YML_PATH" "$SANITIZED_QT_YML" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

drop_prefixes = (
    "prefix:",
    "- torch==",
    "- torchvision==",
    "- torchaudio==",
    "- triton==",
    "- nvidia-",
)

lines = src.read_text(encoding="utf-8").splitlines()
out = []
for line in lines:
    stripped = line.strip()
    if any(stripped.startswith(prefix) for prefix in drop_prefixes):
        continue
    out.append(line)

dst.write_text("\n".join(out) + "\n", encoding="utf-8")
PY

mkdir -p "$REPO_DIR/data/cache" "$REPO_DIR/results/eval"

export MAMBA_ROOT_PREFIX

if [ "$CLEAR_ENV" = "1" ]; then
  if [ -d "$MAMBA_ROOT_PREFIX/envs/$ENV_NAME" ]; then
    "$MICROMAMBA_BIN" env remove -y -n "$ENV_NAME" >/dev/null 2>&1 || rm -rf "$MAMBA_ROOT_PREFIX/envs/$ENV_NAME"
  fi
fi

"$MICROMAMBA_BIN" create -y -n "$ENV_NAME" -f "$SANITIZED_QT_YML"

"$MICROMAMBA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade pip
"$MICROMAMBA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "$TORCH_INDEX_URL"

"$MICROMAMBA_BIN" run -n "$ENV_NAME" python -c 'import os, sys, torch, datasets, transformers, lm_eval, huggingface_hub; print(f"python=={sys.version.split()[0]}"); print(f"torch=={torch.__version__}"); print(f"datasets=={datasets.__version__}"); print(f"transformers=={transformers.__version__}"); print(f"lm_eval=={lm_eval.__version__}"); print(f"huggingface_hub=={huggingface_hub.__version__}"); print(f"cuda_available=={torch.cuda.is_available()}"); print(f"cuda_version=={torch.version.cuda}")'

cat <<EOF

Setup complete.
Repo: $REPO_DIR
Environment: $ENV_NAME
Micromamba: $MICROMAMBA_BIN
Sanitized env file: $SANITIZED_QT_YML
Torch index: $TORCH_INDEX_URL

Activate in zsh/bash with:
  export MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX"
  eval "\$($MICROMAMBA_BIN shell hook -s ${SHELL##*/})"
  micromamba activate "$ENV_NAME"

Run from:
  cd "$REPO_DIR"
EOF
