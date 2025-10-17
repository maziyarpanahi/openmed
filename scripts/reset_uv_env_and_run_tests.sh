#!/usr/bin/env bash

# Reset project environment, recreate a fresh uv virtualenv pinned to Python 3.11,
# install OpenMed (with dev + Hugging Face extras) from source, and run the full
# pytest suite including the slow model-loading scenarios.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

UV_BIN="${UV:-uv}"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "error: uv is not installed or not on PATH" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

echo "[1/5] Removing existing virtual environment and build artifacts"
rm -rf .venv .pytest_cache .mypy_cache build dist .coverage coverage.xml
find . -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "[2/5] Ensuring Python 3.11 toolchain is available to uv"
"${UV_BIN}" python install 3.11 >/dev/null

echo "[3/5] Creating fresh uv-managed virtual environment"
"${UV_BIN}" venv --python 3.11

echo "[4/5] Installing OpenMed (with dev + hf extras) into the new environment"
"${UV_BIN}" pip install --upgrade pip
"${UV_BIN}" pip install --editable '.[dev,hf]'

echo "[5/5] Running pytest suite (slow tests included)"
"${UV_BIN}" run pytest -m "not slow"
"${UV_BIN}" run pytest -m slow --maxfail=1 --disable-warnings

echo "Done. All tests (including slow marked scenarios) completed successfully."
