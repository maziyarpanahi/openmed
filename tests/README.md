# Test Suite

The `tests/` directory contains unit and integration coverage for the OpenMed
library. Tests rely on `pytest` and heavily mock downstream Hugging Face APIs, so
they can be executed offline once the core dependencies are installed.

## Layout

- `unit/` – fast, isolated tests for configuration, model loading helpers,
  tokenisation, formatting, and utility modules.
- `integration/` – higher-level scenarios that exercise the public API surface
  (e.g., `analyze_text`, `list_models`) using mocked transformers pipelines.
- `fixtures/` – shared sample texts and reusable pytest fixtures.
- `conftest.py` – global fixtures for mocking transformers components,
  configuration reset, and sample data.

## Installing from source

Create a fresh virtual environment and install the package in editable mode with
the testing dependencies. The transformer layer is patched during tests but must
still be importable, so install `transformers` (and optionally `torch`).

```bash
# from the repository root
git clone https://github.com/maziyarpanahi/openmed.git
cd openmed
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
pip install pytest pytest-cov transformers
# Optional: install a CPU backend for transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you need to access gated Hugging Face models inside custom tests, export
`HF_TOKEN` before running pytest. The bundled tests do not require network
access.

## Running the tests

Execute the full suite (unit + integration) from the project root:

```bash
pytest
```

Run only the lightweight unit tests:

```bash
pytest tests/unit
# or via markers
pytest -m "not integration"
```

Run only the integration scenarios:

```bash
pytest -m integration
```

Generate coverage reports:

```bash
pytest --cov=openmed --cov-report=term-missing
```

All tests are designed to work offline thanks to extensive mocking, so failures
should either point to regressions in OpenMed code or missing local
requirements.
