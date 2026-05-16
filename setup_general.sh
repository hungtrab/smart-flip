#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
export VENV_DIR="${VENV_DIR:-$REPO_DIR/venv}"
export REQ_FILE="${REQ_FILE:-$REPO_DIR/requirements.txt}"

bash "$SCRIPT_DIR/setup_repo_env.sh"
