#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip >/tmp/pip-up.log
pip install -e '.[dev]' >/tmp/pip-install.log

# Core test suite without slow markers (collect coverage for zero-shot modules)
pytest -m "not slow" --cov=openmed/ner --cov-report=term-missing

# Zero-shot slow checks (gracefully skip when dependencies unavailable)
pytest -m slow
