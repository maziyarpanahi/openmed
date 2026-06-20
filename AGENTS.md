# Repository Guidelines

## Project Structure & Module Organization

Core Python package code lives in `openmed/`. Keep existing privacy and registry code in `openmed/core/`; add genuinely new capability in top-level packages already present or planned: `clinical/`, `eval/`, `multimodal/`, `structured/`, `risk/`, and `interop/`. Backend adapters live in `mlx/`, `coreml/`, and `torch/`; the REST surface is in `service/`; Swift code and demos are in `swift/`. Tests are under `tests/unit`, `tests/integration`, and `tests/fixtures`; docs are in `docs/`; examples are in `examples/`; release and maintenance scripts are in `scripts/`.

## Build, Test, and Development Commands

- `uv pip install -e ".[dev]"`: install editable package with test and lint dependencies.
- `uv pip install -e ".[dev,hf]"`: add Hugging Face dependencies for model-related work.
- `.venv/bin/python -m pytest tests/ -q`: required pre-PR test command for roadmap tasks.
- `pytest --cov=openmed --cov-report=term-missing`: run local coverage.
- `make build`: build wheel and sdist via `python3 -m build`.
- `make docs-serve` / `make docs-build`: preview or strictly build MkDocs.
- `cd swift/OpenMedKit && swift test`: run Swift package tests.

## Coding Style & Architecture Rules

Use Python 3.10+, 4-space indentation, and 88-character lines. Prefer `snake_case` for functions and modules, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Public APIs need Google-style docstrings.

Preserve local-first defaults: no mandatory network calls after model download, no telemetry by default, no raw PHI in logs, caches, temp files, or audit artifacts. Audit reports should use offsets, hashes, provenance, and risk scores rather than plaintext identifiers. Core dependencies must remain permissive-license compatible; GPL, source-available, proprietary, DUA-gated data, UMLS, SNOMED CT, CPT, MIMIC, i2b2, and n2c2 assets must not be bundled. Put restricted integrations behind optional user-supplied keys or out-of-process bridges.

## Testing and Release Gates

Pytest discovers `test_*.py` and `*_test.py`. Mark external or end-to-end cases with `@pytest.mark.integration`; mark expensive checks with `@pytest.mark.slow`. Keep bundled tests offline-friendly through mocks or synthetic fixtures.

For privacy, aggregate F1 is not enough. Add tests or fixtures for direct-identifier recall, critical leakage, span integrity, deterministic safety sweeps, date shifting, surrogate consistency, multilingual IDs, and quantized-model recall deltas when touching those paths. Committed golden data must be synthetic; DUA datasets are eval-only and never committed.

## Commit & Pull Request Guidelines

Recent history uses concise imperative commits, often with prefixes such as `fix:`. For roadmap tasks, keep commits and PRs focused on the task acceptance criteria. Use the repo-owner Git identity. Do not include assistant/tool/vendor names, co-author trailers, generation footers, or worker attribution in issue text, branch names, commit messages, PR titles, PR bodies, or labels. In particular, never use `codex` or `claude` in branch names, PR titles, PR bodies, issue titles, issue bodies, labels, or other repository metadata. Do not assign issues or PRs unless explicitly asked.

Pull requests should follow `.github/PULL_REQUEST_TEMPLATE.md`: description, change type, tests run, docs or changelog updates when applicable, dependency justification, linked issue, and screenshots or example output for UI/docs changes.

## Tracked Files and Commit Safety

Do not add untracked files to git unless the user explicitly asks for those files to be tracked. Never force-add or commit ignored files or ignored directories. Stage only files that are already tracked or files the user has explicitly approved for tracking.

When a file is untracked, ignored, private-looking, generated, or outside the requested scope, leave it alone. If there is any doubt about whether a file should be staged, committed, pushed, or included in a pull request, stop and ask the user before taking the git action.

## OpenMedKit and Multimodal Work

Swift and Python surfaces should evolve together unless a plan explicitly scopes a feature to one platform. OpenMedKit priorities are on-device document intake, OCR, PII redaction, structured extraction, task packs, model catalog/download/cache UX, and later pose, audio, speech, and sensor pipelines. Any borderline or medical-device-style feature must surface disclaimers and must not auto-trigger clinical decisions. Apple Foundation Models paths must stay on-device only; reject any cloud fallback for PHI workflows.
